"""Opt-in causal temporal-padding-to-convolution experiment; no model edits.

Spatial reflection remains explicit. cuDNN handles symmetric temporal zeros;
discard its extra tail outputs to preserve the original left-only causal pad.
This is partial padding absorption, NOT a fully fused Conv3d/SiLU kernel.
"""
import torch
import torch.nn.functional as F


class ImplicitTemporalPadConv3d(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        if not (getattr(original, 'causal', False)
                and getattr(original, 'pad_mode_t', None) == 'constant'
                and not getattr(original, 'spatial_parallel', False)
                and original.groups == 1 and original.dilation == (1, 1, 1)
                and original.padding[0] > 0):
            raise ValueError('Only single-GPU causal zero-temporal-padding convolutions are supported')
        self.weight = original.weight
        self.bias = original.bias
        self.padding = original.padding
        self.stride = original.stride
        self.kernel_size = original.kernel_size
        self.pad_mode = original.pad_mode

    def forward(self, x):
        depth = x.shape[2]
        left = 2 * self.padding[0] if depth > 1 else self.kernel_size[0] - 1
        output_depth = (depth + left - self.kernel_size[0]) // self.stride[0] + 1
        ph, pw = self.padding[1:]
        if ph or pw:
            x = F.pad(x, (pw, pw, ph, ph, 0, 0), mode=self.pad_mode)
        y = F.conv3d(x, self.weight, self.bias, stride=self.stride, padding=(left, 0, 0))
        return y[:, :, :output_depth]


def replace_selected(encoder, names):
    replacements = []
    for name in names:
        parent_path, _, leaf = name.rpartition('.')
        parent = encoder.get_submodule(parent_path) if parent_path else encoder
        original = getattr(parent, leaf)
        setattr(parent, leaf, ImplicitTemporalPadConv3d(original))
        replacements.append((parent, leaf, original))
    return replacements


def restore_selected(replacements):
    for parent, leaf, original in replacements:
        setattr(parent, leaf, original)
