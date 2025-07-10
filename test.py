import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from network import Network
from datagen import *
from scipy.ndimage import binary_fill_holes

import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def dice_score(pred, target, eps=1e-8):
    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    return (2. * intersection + eps) / (union + eps)

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.checkpoint = r'C:\Users\UAB\CK_WorkPlace\PhD\Prostate\sam2dino\weights\proposed\tl\uab_on_xnn\weights\DGSUNet-best.pth'
args.test_image_path = r'D:\PhD\Prostate\Data\samunet\uab\test\images'
args.test_gt_path = r'D:\PhD\Prostate\Data\samunet\uab\test\masks'
args.save_path = r'C:\Users\UAB\CK_WorkPlace\PhD\Prostate\sam2dino\weights\proposed\tl\uab_on_xnn\test'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = DataGen(args.test_image_path, args.test_gt_path, 352,518, mode='test')
model = Network().to(device)
model.load_state_dict(torch.load(args.checkpoint), strict=True)
print('model loaded')
model.eval()
model.cuda()
os.makedirs(args.save_path, exist_ok=True)
dice_scores = []
for image1, image2, gt, name in test_loader:
    with torch.no_grad():
        image1 = image1.to(device)
        image2 = image2.to(device)

        res1, res2, res = model(image2, image1)

        # Normalize prediction
        res = F.interpolate(res, size=(128, 128), mode='bilinear', align_corners=False)
        res = res.sigmoid().cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res_bin = (res > 0.5).astype(np.float32)
        res_bin = binary_fill_holes(res_bin).astype(np.float32)

        # Convert gt to tensor, resize, convert back to numpy
        gt_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
        gt_tensor = F.interpolate(gt_tensor, size=(128, 128), mode='bilinear', align_corners=False)
        gt_np = gt_tensor.squeeze().cpu().numpy()

        # Compute dice
        dice = dice_score(res_bin, gt_np)
        dice_scores.append(dice)
        print(f"Dice for {name}: {dice:.4f}")

        # Save prediction
        save_path = os.path.join(args.save_path, name[:-4] + ".tif")
        imageio.imsave(save_path, (res_bin * 255).astype(np.uint8))