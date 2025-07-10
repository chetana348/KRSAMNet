import torch
import torch.nn as nn
from sam2.build_sam import build_sam2
from matplotlib import rcParams
from utils.feat_extract import transforms_image
#from .feat_extract import visualize_feature_maps_mean, visualize_feature_maps_pca, visualize_feature_maps_tsne


class sam2hiera(nn.Module):
    def __init__(self, config_file=None, ckpt_path=None) -> None:
        super().__init__()
        if config_file is None:
            print("No config file provided, using default config")
            config_file = "./sam2_configs/sam2.1_hiera_l.yaml"
        if ckpt_path is None:
            model = build_sam2(config_file)
        else:
            model = build_sam2(config_file, ckpt_path)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.sam_encoder = model.image_encoder.trunk

        for param in self.sam_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        out = self.sam_encoder(x)
        return out
    