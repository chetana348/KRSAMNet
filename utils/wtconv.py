import pywt
import pywt.data
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F

def _make_2d_filters(vec_a, vec_b):
    """Construct a list of 2D filters via outer product."""
    return [torch.outer(a, b) for a in vec_a for b in vec_b]


def _prepare_coeffs(wavelet_obj, reverse=True, flip=False, dtype=torch.float32):
    """Return low/high filters processed for conv/conv_transpose use."""
    lo = torch.tensor(wavelet_obj.dec_lo if not flip else wavelet_obj.rec_lo, dtype=dtype)
    hi = torch.tensor(wavelet_obj.dec_hi if not flip else wavelet_obj.rec_hi, dtype=dtype)

    if reverse:
        lo = lo.flip(0)
        hi = hi.flip(0)
    
    return lo, hi


def vectorize(wavelet_name, in_ch, out_ch, dtype=torch.float32):
    wavelet = pywt.Wavelet(wavelet_name)

    # Decomposition filters
    lo_d, hi_d = _prepare_coeffs(wavelet, reverse=True, flip=False, dtype=dtype)
    filters_d = _make_2d_filters([lo_d, hi_d], [lo_d, hi_d])
    filters_d = torch.stack(filters_d).unsqueeze(1).repeat(in_ch, 1, 1, 1)

    # Reconstruction filters
    lo_r, hi_r = _prepare_coeffs(wavelet, reverse=True, flip=True, dtype=dtype)
    filters_r = _make_2d_filters([lo_r, hi_r], [lo_r, hi_r])
    filters_r = torch.stack(filters_r).unsqueeze(1).repeat(out_ch, 1, 1, 1)

    return filters_d, filters_r


class ChannelScale(nn.Module):
    def __init__(self, shape, init_scale=1.0, init_bias=None):
        super().__init__()
        self.scale = nn.Parameter(torch.full(shape, init_scale))
        self.bias = nn.Parameter(torch.full(shape, init_bias)) if init_bias is not None else None

    def forward(self, x):
        return self.scale * x + (self.bias if self.bias is not None else 0)


class WaveletResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, stride=1, wavelet_depth=1, wavelet_name='db1', use_bias=True):
        super().__init__()
        self.channels = channels
        self.depth = wavelet_depth
        self.stride = stride

        self.fwd_filter, self.inv_filter = self._init_wavelet_kernels(wavelet_name, channels)
        self.local_conv = self._build_local_conv(channels, kernel_size, use_bias)
        self.local_scale = ChannelScale((1, channels, 1, 1))

        self.freq_convs = nn.ModuleList([
            self._build_freq_conv(channels * 4, kernel_size) for _ in range(wavelet_depth)
        ])
        self.freq_scales = nn.ModuleList([
            ChannelScale((1, channels * 4, 1, 1), init_scale=0.1) for _ in range(wavelet_depth)
        ])

    def forward(self, x):
        features = self._decompose_wavelet(x)
        reconstruction = self._reconstruct_wavelet(features)

        residual = self.local_scale(self.local_conv(x))
        output = reconstruction + residual
        return self._downsample(output) if self.stride > 1 else output

    def _init_wavelet_kernels(self, wavelet_name, channels):
        fwd, inv = vectorize(wavelet_name, channels, channels)
        return nn.Parameter(fwd, requires_grad=False), nn.Parameter(inv, requires_grad=False)

    def _build_local_conv(self, channels, kernel_size, use_bias):
        return nn.Conv2d(channels, channels, kernel_size, padding='same',
                         stride=1, bias=use_bias, groups=channels)

    def _build_freq_conv(self, ch_out, kernel_size):
        return nn.Conv2d(ch_out, ch_out, kernel_size, padding='same',
                         stride=1, groups=ch_out, bias=False)

    def _decompose_wavelet(self, x):
        low_parts, high_parts, shapes = [], [], []
        current = x

        for i in range(self.depth):
            shapes.append(current.shape)
            current = self._pad_if_needed(current)

            encoded = self._encode_wavelet(current)
            current = encoded[:, :, 0, :, :]  # LL only

            freq_in = encoded.reshape(encoded.size(0), -1, encoded.size(-2), encoded.size(-1))
            transformed = self.freq_convs[i](freq_in)
            scaled = self.freq_scales[i](transformed)
            restored = scaled.reshape_as(encoded)

            low_parts.append(restored[:, :, 0, :, :])
            high_parts.append(restored[:, :, 1:, :, :])

        return low_parts, high_parts, shapes

    def _reconstruct_wavelet(self, features):
        low_parts, high_parts, shapes = features
        current = torch.zeros_like(low_parts[-1])

        for i in reversed(range(self.depth)):
            ll = low_parts[i] + current
            hh = high_parts[i]
            ref_shape = shapes[i]

            full_coeffs = torch.cat([ll.unsqueeze(2), hh], dim=2)
            current = self._decode_wavelet(full_coeffs)
            current = current[:, :, :ref_shape[2], :ref_shape[3]]

        return current

    def _encode_wavelet(self, x):
        b, c, h, w = x.shape
        pad_h = self.fwd_filter.size(2) // 2 - 1
        pad_w = self.fwd_filter.size(3) // 2 - 1
        y = F.conv2d(x, self.fwd_filter, stride=2, padding=(pad_h, pad_w), groups=c)
        return y.reshape(b, c, 4, h // 2, w // 2)

    def _decode_wavelet(self, coeff):
        b, c, _, h, w = coeff.shape
        reshaped = coeff.reshape(b, c * 4, h, w)
        pad_h = self.inv_filter.size(2) // 2 - 1
        pad_w = self.inv_filter.size(3) // 2 - 1
        return F.conv_transpose2d(reshaped, self.inv_filter, stride=2, padding=(pad_h, pad_w), groups=c)

    def _pad_if_needed(self, x):
        pad_h = x.size(2) % 2
        pad_w = x.size(3) % 2
        return F.pad(x, (0, pad_w, 0, pad_h)) if pad_h or pad_w else x

    def _downsample(self, x):
        down_weight = torch.ones(self.channels, 1, 1, 1, device=x.device)
        return F.conv2d(x, down_weight, stride=self.stride, groups=self.channels)


def generate_wavelet_kernels(wavelet, in_ch, out_ch, dtype=torch.float32):
    w = pywt.Wavelet(wavelet)

    def create_filters(lo, hi):
        filters = []
        for a in (lo, hi):
            for b in (lo, hi):
                filters.append(torch.outer(a, b))
        return torch.stack(filters).unsqueeze(1)

    lo_d = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    hi_d = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    lo_r = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(0)
    hi_r = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(0)

    filters_d = create_filters(lo_d, hi_d).repeat(in_ch, 1, 1, 1)
    filters_r = create_filters(lo_r, hi_r).repeat(out_ch, 1, 1, 1)

    return filters_d, filters_r


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()

        self.depthwise = WaveletResidualBlock(in_channels, kernel_size=kernel_size)

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
