import numpy as np
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm
from nnunetv2.utilities.network_initialization import InitWeights_He

deep_supervision_scales = [
    [1.0, 1.0, 1.0],
    [0.5, 0.5, 0.5],
    [0.25, 0.25, 0.25],
    # [0.125, 0.125, 0.125],
    # [0.0625, 0.0625, 0.0625]
]
nnunet_weight_factors = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))]).tolist()

def nnunet_configuration(num_segmentation_heads=6, num_input_channels=1):
    # num_input_channels = 1
    num_stages = 6
    UNet_base_num_features = 32
    unet_max_num_features = 320
    conv_op = nn.Conv3d
    conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    pool_op_kernel_sizes = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]]
    # num_segmentation_heads = 6
    deep_supervision = True
    conv_or_blocks_per_stage = {
        'n_conv_per_stage': [2, 2, 2, 2, 2, 2], 
        'n_conv_per_stage_decoder': [2, 2, 2, 2, 2]
    }
    kwargs = {
        'conv_bias': True,
        'norm_op': get_matching_instancenorm(conv_op),
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        # 'dropout_op': None, 'dropout_op_kwargs': None,
        'dropout_op': nn.Dropout3d, 'dropout_op_kwargs': {'p': 0.2},
        'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
    }
    model = PlainConvUNet(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(UNet_base_num_features * 2 ** i,
                                unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=conv_kernel_sizes,
        strides=pool_op_kernel_sizes,
        num_classes=num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs
    )
    model.apply(InitWeights_He(1e-2))
    return model

class MultiHeadnnUNet(nn.Module):
    def __init__(self, num_segmentation_heads, num_input_channels):
        super().__init__()
        self.model = nnunet_configuration(1, num_input_channels)
        input_features_skip = [m.in_channels for m in self.model.decoder.seg_layers]
        self.model.decoder.seg_layers = nn.ModuleList([nn.Identity() for _ in range(len(self.model.decoder.seg_layers))])

        conv_op = nn.Conv3d
        seg_layers = {}
        for k, v in num_segmentation_heads.items():
            seg_layers[k] = nn.ModuleList([
                conv_op(in_ch, v, 1, 1, 0, bias=True) for in_ch in input_features_skip
            ])
        self.seg_layers = nn.ModuleDict(seg_layers)
    
    def forward(self, x):
        x = self.model(x)
        out = {}
        for k, m in self.seg_layers.items():
            out[k] = [mm(xx) for mm, xx in zip(m[::-1], x)]
        # x = {k: m(x)[::-1] for xx, k, m in zip(x, self.seg_layers.items())}
        return out, None # needed for compatibility with pm_cls