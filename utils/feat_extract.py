import torch
import torch.nn as nn
import numpy as np
from matplotlib import rcParams
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
from matplotlib import rcParams
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torchvision.transforms as transforms


def transforms_image(image_path, image_size=518):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)
    
    
class FeatureExtractor(nn.Module):
    def __init__(self, variant=None, repo=None):
        super().__init__()
        self.source = repo or 'facebookresearch/dinov2'
        self.model_id = variant or 'dinov2_vitl14'

        self.encoder = torch.hub.load(self.source, self.model_id, pretrained=True)
        self.patch_size = 14

        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, image_tensor):
        result = self.encoder.forward_features(image_tensor)
        tokens = result['x_norm_patchtokens']

        B = image_tensor.size(0)
        H = image_tensor.size(-1)
        F = int((H / self.patch_size) ** 2)
        S = int(np.sqrt(F))

        token_map = tokens.view(B, S, S, -1).permute(0, 3, 1, 2)
        return token_map
        

def plot_mean_activation_maps(feature_dict, model_name="default"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    levels = ['top_level', 'high_level', 'mid_level', 'low_level']
    titles = ['Level3 Feature Map', 'Level2 Feature Map', 'Level1 Feature Map', 'Level0 Feature Map']

    for idx, (ax, level, title) in enumerate(zip(axes, levels, titles)):
        feat_tensor = feature_dict[level].detach().cpu()
        avg_map = torch.mean(feat_tensor, dim=1)[0].numpy()
        im = ax.imshow(avg_map, cmap='viridis')
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    fig.suptitle('Mean Feature Map Visualization', fontsize=16)
    plt.tight_layout()
    save_path = f'../visualize/pic/{model_name}_multiscale_features_mean.png'
    plt.savefig(save_path)
    plt.show()

def plot_tsne_projection(feature_maps, model_id="default"):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    levels = ["top_level", "high_level", "mid_level", "low_level"]
    titles = ["Level3 Feature Map", "Level2 Feature Map", "Level1 Feature Map", "Level0 Feature Map"]
    colors = ["blue", "blue", "green", "red"]

    for idx, (ax, level, title, color) in enumerate(zip(axes, levels, titles, colors)):
        fmap = feature_maps[level].detach().cpu().numpy()[0]
        flattened = fmap.reshape(fmap.shape[0], -1).T  # (C, H*W) -> (H*W, C)
        embedding = TSNE(n_components=2, random_state=42).fit_transform(flattened)
        ax.scatter(embedding[:, 0], embedding[:, 1], c=color, s=1)
        ax.set_title(title)
        fig.colorbar(ax.collections[0], ax=ax)

    fig.suptitle("T-SNE Feature Map Visualization", fontsize=16)
    plt.tight_layout()
    save_path = f"../visualize/pic/{model_id}_multiscale_features_tsne.png"
    plt.savefig(save_path)
    plt.show()


def plot_pca_feature_projection(feature_set, model_name="default"):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import numpy as np

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    stages = ["top_level", "high_level", "mid_level", "low_level"]
    labels = ["Level3 Feature Map", "Level2 Feature Map", "Level1 Feature Map", "Level0 Feature Map"]

    for idx, (ax, key, label) in enumerate(zip(axes, stages, labels)):
        fmap = feature_set[key].detach().cpu().numpy()[0]  # (C, H, W)
        reshaped = fmap.reshape(fmap.shape[0], -1)  # (C, H*W)
        reduced = PCA(n_components=1).fit_transform(reshaped.T).flatten()  # (H*W,)
        side_len = int(np.sqrt(len(reduced)))
        viewable = reduced.reshape(side_len, -1)

        im = ax.imshow(viewable, cmap='viridis')
        ax.set_title(label)
        fig.colorbar(im, ax=ax)

    fig.suptitle("PCA Feature Map Visualization", fontsize=16)
    plt.tight_layout()
    save_name = f"../visualize/pic/{model_name}_multiscale_features_pca.png"
    plt.savefig(save_name)
    plt.show()