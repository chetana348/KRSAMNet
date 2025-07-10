import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange
#from torchinfo import summary
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedSpatialMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.refine = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7,
                                stride=1, padding=3, padding_mode='reflect')

    def forward(self, inp):
        pooled = torch.stack([inp.mean(1, keepdim=True), inp.amax(1, keepdim=True)], dim=1)
        mix = pooled.view(inp.size(0), 2, inp.size(2), inp.size(3))
        return self.refine(mix)


class ContextChannelScaler(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        bottleneck = channels // reduction_ratio
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.reduction = nn.Conv2d(channels, bottleneck, kernel_size=1)
        self.restore = nn.Conv2d(bottleneck, channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, feat):
        squeeze = self.pool(feat)
        excite = self.act(self.reduction(squeeze))
        return self.restore(excite)


class FeatureGating(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.conv = nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=7, padding=3,
                              groups=feat_dim, padding_mode='reflect')
        self.norm = nn.Sigmoid()

    def forward(self, primary, guidance):
        concat = torch.cat([primary.unsqueeze(2), guidance.unsqueeze(2)], dim=2)
        layout = rearrange(concat, 'b c t h w -> b (c t) h w')
        return self.norm(self.conv(layout))


class MultiAttn(nn.Module):
    def __init__(self, feat_size, shrink=8):
        super().__init__()
        self.spatial_mod = MixedSpatialMap()
        self.channel_mod = ContextChannelScaler(feat_size, shrink)
        self.gate_mod = FeatureGating(feat_size)
        self.output = nn.Conv2d(feat_size, feat_size, kernel_size=1)

    def forward(self, input_a, input_b):
        sum_feats = input_a + input_b
        spatial_focus = self.spatial_mod(sum_feats)
        channel_focus = self.channel_mod(sum_feats)
        blend_mask = self.gate_mod(sum_feats, spatial_focus + channel_focus)
        fused = sum_feats + blend_mask * input_a + (1.0 - blend_mask) * input_b
        return self.output(fused)
        


class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.norm_relu = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.norm_relu(x)


class SelfAttnUnit(nn.Module):
    def __init__(self, in_key, in_query, hidden_dim, out_dim, n_kq, n_vo):
        super().__init__()
        self.k_proj = self._build_transform(in_key, hidden_dim, n_kq)
        self.q_proj = self._build_transform(in_query, hidden_dim, n_kq)
        self.v_proj = self._build_transform(in_key, hidden_dim, n_vo)
        self.o_proj = self._build_transform(hidden_dim, out_dim, n_vo)
        self.scale = hidden_dim ** -0.5

    def _build_transform(self, ch_in, ch_out, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(ch_in if i == 0 else ch_out, ch_out, 1, bias=False))
            layers.append(nn.BatchNorm2d(ch_out))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, q_input, k_input, v_input):
        B = q_input.shape[0]
        q = self.q_proj(q_input).flatten(2).transpose(1, 2)  # (B, N, C)
        k = self.k_proj(k_input).flatten(2)  # (B, C, N)
        v = self.v_proj(v_input).flatten(2).transpose(1, 2)  # (B, N, C)

        sim = torch.matmul(q, k) * self.scale
        attn = F.softmax(sim, dim=-1)

        context = torch.matmul(attn, v).transpose(1, 2).reshape(B, -1, *q_input.shape[2:])
        return self.o_proj(context)


class Decoder(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.low_res_transform = BasicConv(feature_dim, feature_dim, k_size=1)
        self.high_res_transform = BasicConv(feature_dim, feature_dim, k_size=1)
        self.fuse = BasicConv(feature_dim * 2, feature_dim, k_size=3, padding=1)
        self.attn = SelfAttnUnit(
            in_key=feature_dim,
            in_query=feature_dim,
            hidden_dim=feature_dim // 2,
            out_dim=feature_dim,
            n_kq=2,
            n_vo=1
        )

    def forward(self, feat_small, feat_large):
        upsampled = F.interpolate(feat_small, size=feat_large.shape[2:], mode='bilinear', align_corners=False)
        enhanced = self.low_res_transform(upsampled) + self.high_res_transform(feat_large)
        context = self.attn(enhanced, enhanced, feat_large)
        return self.fuse(torch.cat([context, feat_large], dim=1))



class TFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TFF, self).__init__()
        self.catconvA = dsconv_3x3(in_channel * 2, in_channel)
        self.catconvB = dsconv_3x3(in_channel * 2, in_channel)
        self.catconv = dsconv_3x3(in_channel * 2, out_channel)
        self.convA = nn.Conv2d(in_channel, 1, 1)
        self.convB = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xA, xB):
        x_diff = xA - xB

        x_diffA = self.catconvA(torch.cat([x_diff, xA], dim=1))
        x_diffB = self.catconvB(torch.cat([x_diff, xB], dim=1))

        A_weight = self.sigmoid(self.convA(x_diffA))
        B_weight = self.sigmoid(self.convB(x_diffB))

        xA = A_weight * xA
        xB = B_weight * xB

        x = self.catconv(torch.cat([xA, xB], dim=1))

        return x

