import os
import sys
sys.path.append(r"C:\Users\UAB\CK_WorkPlace\PhD\Prostate\sam2dino")
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from datagen import *
from tqdm import tqdm
from network import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def structure_loss(pred, mask):
    def _weighted_bce(logit, target, weight):
        bce = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
        return (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    def _weighted_iou(pred, target, weight):
        pred = torch.sigmoid(pred)
        inter = ((pred * target) * weight).sum(dim=(2, 3))
        union = ((pred + target) * weight).sum(dim=(2, 3))
        return 1 - (inter + 1) / (union - inter + 1)

    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, stride=1, padding=15) - mask)
    return (_weighted_bce(pred, mask, weight) + _weighted_iou(pred, mask, weight)).mean()


def compute_dice_score(pred, target, threshold=0.5, eps=1e-8):
    pred = torch.sigmoid(pred)
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    intersection = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def _setup_optimizer(model, lr, weight_decay):
    return opt.AdamW([{"params": model.parameters(), "initia_lr": lr}], lr=lr, weight_decay=weight_decay)


def _train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    print(f'\n[Epoch {epoch+1}] Training...')
    epoch_loss = 0.0
    epoch_dice = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        x1 = batch['image1'].to(device)
        x2 = batch['image2'].to(device)
        target = batch['label'].to(device)

        optimizer.zero_grad()
        pred0, pred1, pred2 = model(x2, x1)
        loss = 0.25 * structure_loss(pred0, target) + 0.5 * structure_loss(pred1, target) + structure_loss(pred2, target)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            dice = compute_dice_score(pred2, target)

        epoch_loss += loss.item()
        epoch_dice += dice

    return epoch_loss / len(dataloader), epoch_dice / len(dataloader)

def _validate(model, dataloader, device):
    model.eval()
    val_dice = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            x1 = batch['image1'].to(device)
            x2 = batch['image2'].to(device)
            target = batch['label'].to(device)
            _, _, pred = model(x2, x1)
            val_dice += compute_dice_score(pred, target)
    return val_dice / len(dataloader)

def _save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print("[Saved model to]", path)


def main(args):
    os.makedirs(args.save_path, exist_ok=True)

    train_dataset = DataGen(args.train_image_path, args.train_mask_path, 352, 518, mode='train')
    val_dataset = DataGen(args.val_image_path, args.val_mask_path, 352, 518, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network(args.dino_model_name, args.dino_hub_dir, args.sam_config_file, args.sam_ckpt_path).to(device)
    model.load_state_dict(torch.load(r'C:\Users\UAB\CK_WorkPlace\PhD\Prostate\sam2dino\weights\proposed\base\x\weights\DGSUNet-best.pth'), strict=True)
    print('X Model Loaded')
    optimizer = _setup_optimizer(model, args.lr, args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=1e-7)

    best_val_dice = 0.0
    for epoch in range(args.epoch):
        train_loss, train_dice = _train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_dice = _validate(model, val_loader, device)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}")
        scheduler.step()

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_path = os.path.join(args.save_path, f'DGSUNet-best.pth')
            _save_checkpoint(model, best_path)




if __name__ == "__main__":
    # seed_torch(1024)
    args = argparse.ArgumentParser()
    args.dino_model_name = "dinov2_vitl14"
    args.dino_hub_dir = "facebookresearch/dinov2"
    args.sam_config_file = "sam2.1_hiera_l.yaml"
    args.sam_ckpt_path = r"C:\Users\UAB\CK_WorkPlace\PhD\Prostate\sam2dino\sam2.1_hiera_large.pt"
    args.train_image_path = r'D:\PhD\Prostate\Data\samunet\uab\train\images'
    args.train_mask_path = r'D:\PhD\Prostate\Data\samunet\uab\train\masks'
    args.val_image_path = r'D:\PhD\Prostate\Data\samunet\uab\test\images'
    args.val_mask_path = r'D:\PhD\Prostate\Data\samunet\uab\test\masks'
    
    args.save_path = r'C:\Users\UAB\CK_WorkPlace\PhD\Prostate\sam2dino\weights\proposed\tl\uab_on_xnn\weights'
    args.epoch = 20
    args.lr = 0.001
    args.batch_size = 2
    args.weight_decay = 5e-4
    main(args)