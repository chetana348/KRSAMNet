import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import feat_extract
from utils import samhiera
from utils.multattn import *
from utils import wtconv
#from torchinfo import summary

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class ContextualPyramidBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ContextualPyramidBlock, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class Network(nn.Module):
    def __init__(self, dino_model_name=None, dino_hub_dir=None, sam_config_file=None, sam_ckpt_path=None):
        super().__init__()
        # Set defaults
        dino_model_name, dino_hub_dir, sam_config_file, sam_ckpt_path = self._set_defaults(
            dino_model_name, dino_hub_dir, sam_config_file, sam_ckpt_path
        )

        # Initialize network modules
        self._init_backbones(dino_model_name, dino_hub_dir, sam_config_file, sam_ckpt_path)
        self._init_fusion_and_heads()
        self._init_decoders()

    def forward(self, x_dino, x_sam):
        # Feature extraction
        feats_sam, feat_dino = self._extract_features(x_sam, x_dino)

        # Fusion and decoding
        out1, out2, out3 = self._fuse_and_decode(feats_sam, feat_dino)
        return out1, out2, out3

    def _set_defaults(self, dino_model_name, dino_hub_dir, sam_config_file, sam_ckpt_path):
        if dino_model_name is None:
            print("No model_name specified, using default")
            dino_model_name = 'dinov2_vitl14'
        if dino_hub_dir is None:
            print("No dino_hub_dir specified, using default")
            dino_hub_dir = 'facebookresearch/dinov2'
        if sam_config_file is None:
            print("No sam_config_file specified, using default")
            sam_config_file = 'sam2.1_hiera_l.yaml'
        if sam_ckpt_path is None:
            print("No sam_ckpt_path specified, using default")
            sam_ckpt_path = 'sam2.1_hiera_large.pt'
        return dino_model_name, dino_hub_dir, sam_config_file, sam_ckpt_path

    def _init_backbones(self, model_name, hub_dir, config_file, ckpt_path):
        self.backbone_dino = feat_extract.FeatureExtractor(model_name, hub_dir)
        self.backbone_sam = samhiera.sam2hiera(config_file, ckpt_path)

    def _init_fusion_and_heads(self):
        self.fusion = MultiAttn(1152)
        self.align_dino = wtconv.ConvBlock(1024, 1152)

        self.rfb_blocks = nn.ModuleList([
            ContextualPyramidBlock(144, 64),
            ContextualPyramidBlock(288, 64),
            ContextualPyramidBlock(576, 64),
            ContextualPyramidBlock(1152, 64)
        ])

    def _init_decoders(self):
        self.decoders = nn.ModuleList([
            Decoder(64),
            Decoder(64),
            Decoder(64)
        ])
        self.side_heads = nn.ModuleList([
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Conv2d(64, 1, kernel_size=1)
        ])
        self.final_head = nn.Conv2d(64, 1, kernel_size=1)

    def _extract_features(self, x_sam, x_dino):
        x1, x2, x3, x4 = self.backbone_sam(x_sam)
        dino_feat = self.backbone_dino(x_dino)
        return (x1, x2, x3, x4), dino_feat

    def _fuse_and_decode(self, sam_feats, dino_feat):
        x1, x2, x3, x4 = sam_feats

        # Resize & align dino to sam resolution
        dino_resized = F.interpolate(dino_feat, size=(11, 11), mode='bilinear', align_corners=False)
        dino_aligned = self.align_dino(dino_resized)

        # Fuse features
        fused = self.fusion(x4, dino_aligned)

        # RFB pyramid features
        x1, x2, x3, x4 = (
            self.rfb_blocks[0](x1),
            self.rfb_blocks[1](x2),
            self.rfb_blocks[2](x3),
            self.rfb_blocks[3](fused)
        )

        # Decoder stages
        x = self.decoders[0](x4, x3)
        out1 = F.interpolate(self.side_heads[0](x), scale_factor=16, mode='bilinear')

        x = self.decoders[1](x, x2)
        out2 = F.interpolate(self.side_heads[1](x), scale_factor=8, mode='bilinear')

        x = self.decoders[2](x, x1)
        out3 = F.interpolate(self.final_head(x), scale_factor=4, mode='bilinear')

        return out1, out2, out3

######################################################################################################


