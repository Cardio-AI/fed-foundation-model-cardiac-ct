import os
from copy import deepcopy
from pathlib import Path
from typing import Union
from collections.abc import Sequence
import itertools

import sys
sys.path.append('/mnt/ssd/git-repos/fedbiomed')
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.data import DataManager, MedicalFolderDataset

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.networks.blocks import UnetOutBlock
from monai.data.itk_torch_bridge import metatensor_to_itk_image
from monai.inferers import sliding_window_inference
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep
import torchio as tio
import itk
import SimpleITK as sitk
from scipy import ndimage
import cc3d
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ImageNormalization
from nnunetv2.utilities.network_initialization import InitWeights_He


class TP(TorchTrainingPlan):
    X_KEY = 'Heart ROI New Origin'
    Y_NEAREST_KEYS = {
        'Heart Seg New Origin': 'heart',
        'Calc New Origin': 'calc'
    }
    Y_PTS_KEYS = {
        'HPS New Origin': 'hps',
        'MS New Origin': 'ms',
    }
    Y_BILINEAR_KEYS= {
        'Logits HPS': 'logits_hps',
        'Logits MS': 'logits_ms',
        'Logits Calc': 'logits_calc',
        'Uncertainty HPS': 'uncert_hps',
        'Uncertainty MS': 'uncert_ms',
        'Uncertainty Calc': 'uncert_calc',
    }
    Y_TORCH_KEYS = {
        'Logits HPS Torch': 'logits_hps', 
        'Logits MS Torch': 'logits_ms', 
        'Logits Calc Torch': 'logits_calc'
    }

    fed_task_to_y_key = {
        1: ['HP', 'hps'],
        2: ['MS', 'ms'],
        3: ['Calcification focused', 'calc']
    }

    key_to_labels = {
        'heart': ['Myo', 'LA', 'LV', 'RA', 'RV', 'Aorta', 'PA'],
        'hps': ['RCC', 'LCC', 'ACC', 'RCA', 'LCA'],
        'ms': ['MS1', 'MS2'],
        'calc': ['Calc']
    }
    criterion = DC_and_CE_loss(ce_kwargs={},soft_dice_kwargs={'batch_dice': True, 'do_bg': False, 'smooth': 1., 'ddp': False})
    
    intensityproperties = {
        "mean": -439.6132772, 
        "median": -364.0, 
        "std": 520.9826361876911, 
        "min": -1024.0, 
        "max": 3071.0, 
        "percentile_99_5": 696.0, 
        "percentile_00_5": -1024.0
    }
    radius = 4
    target_spacing = (1.0,1.0,1.0)
    target_resolution = (192,192,192)
    patch_size = (96, 96, 96)
    DEBUG = True

    deep_supervision_scales = [
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
        [0.25, 0.25, 0.25],
        # [0.125, 0.125, 0.125],
        # [0.0625, 0.0625, 0.0625]
    ]
    weight_factors = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))]).tolist()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = self.Transform(
            intensityproperties=self.intensityproperties,
            patch_size=self.patch_size,
            target_spacing=self.target_spacing,
            radius=self.radius
        )

    def init_dependencies(self):
        deps = ['import torch',
                "import torch.nn as nn",
                'import torch.nn.functional as F',
                "from fedbiomed.common.data import MedicalFolderDataset",
                'from monai.networks.nets.swin_unetr import SwinUNETR',
                'from monai.networks.blocks import UnetOutBlock',
                'from monai.data.itk_torch_bridge import metatensor_to_itk_image',
                'from monai.inferers import sliding_window_inference',
                'from monai.networks.blocks.dynunet_block import UnetOutBlock',
                'from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock',
                'from monai.networks.nets.vit import ViT',
                'from monai.utils import ensure_tuple_rep',
                'from typing import Union',
                'from collections.abc import Sequence',
                'import os',
                'import itertools',
                'import numpy as np',
                'from copy import deepcopy',
                'import itk',
                'import SimpleITK as sitk',
                'from scipy import ndimage',
                'import cc3d',
                'import torchio as tio',
                'from pathlib import Path',
                'from dynamic_network_architectures.architectures.unet import PlainConvUNet',
                'from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm',
                'from nnunetv2.preprocessing.normalization.default_normalization_schemes import ImageNormalization',
                # 'from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper',
                'from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss',
                'from nnunetv2.training.loss.compound_losses import DC_and_CE_loss',
                'from nnunetv2.training.loss.dice import get_tp_fp_fn_tn',
                'from nnunetv2.utilities.network_initialization import InitWeights_He',
                ]
        return deps

    class MultiHeadSwinUNETR(nn.Module):
        def __init__(
                self, 
                model_args={},
                # out_channels={'heart': 8, 'hps': 6, 'ms': 3, 'calc': 2},
                # img_size=(96,96,96), 
                # in_channels=1, 
                intermediate_out_channels=1,
                ckpt_path=None
            ):
            super().__init__()
            spatial_dims = 3
            feature_size = 48

            in_channels = model_args.get('in_channels', 1)
            out_channels = model_args.get('out_channels', {'heart': 8, 'hps': 6, 'ms': 3, 'calc': 2})
            fed_task = model_args.get('fed_task', 0)
            finetune = model_args.get('finetune', False)
            img_size = model_args.get('img_size', (96,96,96))
            patches = model_args.get('patches', 0.)
            deep_supervision = model_args.get('deep_supervision', 1.)
            condition_on_seg = model_args.get('condition_on_seg', 1.)
            output_seg = model_args.get('output_seg', 0.)
            self.register_buffer('fed_task', torch.tensor(float(fed_task)))
            self.register_buffer('patches', torch.tensor(float(patches)))
            self.register_buffer('deep_supervision', torch.tensor(float(deep_supervision)))
            self.register_buffer('condition_on_seg', torch.tensor(float(condition_on_seg)))
            self.register_buffer('output_seg', torch.tensor(float(output_seg)))

            self.swin_unetr = SwinUNETR(
                img_size=img_size, # (96,96,96),
                in_channels=in_channels, # 1,
                out_channels=intermediate_out_channels, # 1,
                feature_size=feature_size,
                use_checkpoint=True,
            )#.cuda()
            if ckpt_path is not None:
                self.swin_unetr.load_from(
                    weights=torch.load(ckpt_path) # "./checkpoints/model_swinvit.pt")
                )

            block = lambda oc: UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=oc)
            self.outs = nn.ModuleDict({n: block(oc) for n, oc in out_channels.items()})

            if finetune:
                for param in self.swin_unetr.parameters():
                    param.requires_grad = False

            # try:
            if 'CKPT' in os.environ:
                ckpt = torch.load(os.environ['CKPT'], map_location='cpu') # 'experiments/federated/hps-conditioned-on-heart/1696572867.003709/final_ckpt.pt')
                # ckpt['fed_task'] = self.fed_task
                # ckpt['patches'] = self.patches
                # ckpt['deep_supervision'] = self.deep_supervision
                # ckpt['condition_on_seg'] = self.condition_on_seg
                # ckpt['output_seg'] = self.output_seg
                for k, v in self.state_dict().items():
                    if k not in ckpt:
                        ckpt[k] = v
                ckpt['fed_task'] = self.fed_task
                self.load_state_dict(ckpt)
                print('LOADING CKPT SUCCESFUL')
            # except:
            #     print('NOT ON HOST MACHINE')

        def forward(self, x_in):
            hidden_states_out = self.swin_unetr.swinViT(x_in, self.swin_unetr.normalize)
            enc0 = self.swin_unetr.encoder1(x_in)
            enc1 = self.swin_unetr.encoder2(hidden_states_out[0])
            enc2 = self.swin_unetr.encoder3(hidden_states_out[1])
            enc3 = self.swin_unetr.encoder4(hidden_states_out[2])
            dec4 = self.swin_unetr.encoder10(hidden_states_out[4])
            dec3 = self.swin_unetr.decoder5(dec4, hidden_states_out[3])
            dec2 = self.swin_unetr.decoder4(dec3, enc3)
            dec1 = self.swin_unetr.decoder3(dec2, enc2)
            dec0 = self.swin_unetr.decoder2(dec1, enc1)
            out = self.swin_unetr.decoder1(dec0, enc0)

            out = {n: m(out) for n, m in self.outs.items()}
            # out['hidden_states'] = hidden_states_out

            return out

    @staticmethod
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
        def __init__(
                self, 
                model_args={},
            ):
            super().__init__()
            in_channels = model_args.get('in_channels', 1)
            out_channels = model_args.get('out_channels', {'heart': 8, 'hps': 6, 'ms': 3, 'calc': 2})
            fed_task = model_args.get('fed_task', 0)
            finetune = model_args.get('finetune', False)
            patches = model_args.get('patches', 0.)
            deep_supervision = model_args.get('deep_supervision', 1.)
            condition_on_seg = model_args.get('condition_on_seg', 1.)
            output_seg = model_args.get('output_seg', 0.)
            self.register_buffer('fed_task', torch.tensor(float(fed_task)))
            self.register_buffer('patches', torch.tensor(float(patches)))
            self.register_buffer('deep_supervision', torch.tensor(float(deep_supervision)))
            self.register_buffer('condition_on_seg', torch.tensor(float(condition_on_seg)))
            self.register_buffer('output_seg', torch.tensor(float(output_seg)))

            self.model = TP.nnunet_configuration(1, in_channels)
            input_features_skip = [m.in_channels for m in self.model.decoder.seg_layers]
            self.model.decoder.seg_layers = nn.ModuleList([nn.Identity() for _ in range(len(self.model.decoder.seg_layers))])

            conv_op = nn.Conv3d
            seg_layers = {}
            for k, v in out_channels.items():
                seg_layers[k] = nn.ModuleList([
                    conv_op(in_ch, v, 1, 1, 0, bias=True) for in_ch in input_features_skip
                ])
            self.seg_layers = nn.ModuleDict(seg_layers)

            if finetune:
                for param in self.model.parameters():
                    param.requires_grad = False
            
            if 'CKPT' in os.environ:
                ckpt = torch.load(os.environ['CKPT'], map_location='cpu')
                if 'model' in ckpt:
                    ckpt = ckpt['model']
                # ckpt['fed_task'] = self.fed_task
                # ckpt['patches'] = self.patches
                # ckpt['deep_supervision'] = self.deep_supervision
                # ckpt['condition_on_seg'] = self.condition_on_seg
                # ckpt['output_seg'] = self.output_seg
                
                ckpt = {k.replace('unet.', 'model.'): v for k, v in ckpt.items()}
                for k, v in self.state_dict().items():
                    if k not in ckpt:
                        ckpt[k] = v
                ckpt['fed_task'] = self.fed_task

                self.load_state_dict(ckpt)
                print('LOADING CKPT SUCCESFUL')
        
        def forward(self, x):
            x = self.model(x)
            out = {}
            for k, m in self.seg_layers.items():
                out[k] = [mm(xx) for mm, xx in zip(m[::-1], x)]
            return out
    
    class MultiHeadUNETR(nn.Module):
        def __init__(
            self,
            model_args={},
            # in_channels: int,
            # out_channels: Dict[str, int],
            # img_size: Sequence[int] | int,
            feature_size: int = 16,
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_heads: int = 12,
            pos_embed: str = "conv",
            proj_type: str = "conv",
            norm_name: Union[tuple, str] = "instance",
            conv_block: bool = True,
            res_block: bool = True,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
            qkv_bias: bool = False,
            save_attn: bool = False,

            # with_skip: bool = True,
            # with_classification: bool = False,
            # num_classes: int = 2,
            # cls_heads: Optional[Dict[str, int]] = None
        ) -> None:
            """
            Args:
                in_channels: dimension of input channels.
                out_channels: dimension of output channels.
                img_size: dimension of input image.
                feature_size: dimension of network feature size. Defaults to 16.
                hidden_size: dimension of hidden layer. Defaults to 768.
                mlp_dim: dimension of feedforward layer. Defaults to 3072.
                num_heads: number of attention heads. Defaults to 12.
                proj_type: patch embedding layer type. Defaults to "conv".
                norm_name: feature normalization type and arguments. Defaults to "instance".
                conv_block: if convolutional block is used. Defaults to True.
                res_block: if residual block is used. Defaults to True.
                dropout_rate: fraction of the input units to drop. Defaults to 0.0.
                spatial_dims: number of spatial dims. Defaults to 3.
                qkv_bias: apply the bias term for the qkv linear layer in self attention block. Defaults to False.
                save_attn: to make accessible the attention in self attention block. Defaults to False.

            .. deprecated:: 1.4
                ``pos_embed`` is deprecated in favor of ``proj_type``.

            Examples::

                # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
                >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

                # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
                >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)

                # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
                >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), proj_type='conv', norm_name='instance')

            """

            super().__init__()

            if not (0 <= dropout_rate <= 1):
                raise ValueError("dropout_rate should be between 0 and 1.")

            if hidden_size % num_heads != 0:
                raise ValueError("hidden_size should be divisible by num_heads.")

            in_channels = model_args.get('in_channels', 1)
            out_channels = model_args.get('out_channels', {'heart': 8, 'hps': 6, 'ms': 3, 'calc': 2})
            fed_task = model_args.get('fed_task', 0)
            finetune = model_args.get('finetune', False)
            img_size = model_args.get('img_size', (192,192,192))
            patches = model_args.get('patches', 0.)
            deep_supervision = model_args.get('deep_supervision', 1.)
            condition_on_seg = model_args.get('condition_on_seg', 1.)
            output_seg = model_args.get('output_seg', 0.)
            self.register_buffer('fed_task', torch.tensor(float(fed_task)))
            self.register_buffer('patches', torch.tensor(float(patches)))
            self.register_buffer('deep_supervision', torch.tensor(float(deep_supervision)))
            self.register_buffer('condition_on_seg', torch.tensor(float(condition_on_seg)))
            self.register_buffer('output_seg', torch.tensor(float(output_seg)))

            self.num_layers = 12
            img_size = ensure_tuple_rep(img_size, spatial_dims)
            self.patch_size = ensure_tuple_rep(16, spatial_dims)
            self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
            self.hidden_size = hidden_size

            self.vit = ViT(
                in_channels=in_channels,
                img_size=img_size,
                patch_size=self.patch_size,
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_layers=self.num_layers,
                num_heads=num_heads,
                proj_type=proj_type,
                classification=False,
                dropout_rate=dropout_rate,
                spatial_dims=spatial_dims,
                qkv_bias=qkv_bias,
                save_attn=save_attn,
            )

            self.encoder1 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.encoder2 = UnetrPrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=feature_size * 2,
                num_layer=2,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
            )
            self.encoder3 = UnetrPrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=feature_size * 4,
                num_layer=1,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
            )
            self.encoder4 = UnetrPrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=feature_size * 8,
                num_layer=0,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
            )
            self.decoder5 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=feature_size * 8,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder4 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 8,
                out_channels=feature_size * 4,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder3 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 4,
                out_channels=feature_size * 2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder2 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 2,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )

            # self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
            out = lambda c: UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=c)
            self.outs = nn.ModuleDict({
                k: out(c) for k, c in out_channels.items()
            })
            self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
            self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

            backbones = [
                self.vit, self.encoder1, self.encoder2, self.encoder3, self.encoder4,
                self.decoder2, self.decoder3, self.decoder4, self.decoder5
            ]
            if finetune:
                for m in backbones:
                    for p in m.parameters():
                        p.requires_grad = False
            
            if 'CKPT' in os.environ:
                ckpt = torch.load(os.environ['CKPT'])
                ckpt['fed_task'] = self.fed_task
                self.load_state_dict(ckpt)
                print('LOADING CKPT SUCCESFUL')

        def proj_feat(self, x):
            new_view = [x.size(0)] + self.proj_view_shape
            x = x.view(new_view)
            x = x.permute(self.proj_axes).contiguous()
            return x

        def forward(self, x_in):
            x, hidden_states_out = self.vit(x_in)
            # if self.with_classification:
            #     out_cls = self.classification_head(x[:,0])
            #     x = x[:,1:]
            #     hidden_states_out = [h[:,1:] for h in hidden_states_out]
            # else:
            #     out_cls = None
            # if self.with_skip:
            enc1 = self.encoder1(x_in)
            x2 = hidden_states_out[3]
            enc2 = self.encoder2(self.proj_feat(x2))
            x3 = hidden_states_out[6]
            enc3 = self.encoder3(self.proj_feat(x3))
            x4 = hidden_states_out[9]
            enc4 = self.encoder4(self.proj_feat(x4))

            dec4 = self.proj_feat(x)
            dec3 = self.decoder5(dec4, enc4)
            dec2 = self.decoder4(dec3, enc3)
            dec1 = self.decoder3(dec2, enc2)
            out = self.decoder2(dec1, enc1)

            out_seg = {k: o(out) for k, o in self.outs.items()}
            return out_seg # , out_cls
    
    class CTNormalization(ImageNormalization):
        leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

        @staticmethod
        def type_to_clip_fn(img):
            if type(img) == np.ndarray:
                return np.clip
            elif type(img) == torch.Tensor:
                return torch.clamp
            elif type(img) == sitk.Image:
                return lambda x, lb, ub: sitk.Clamp(x, lowerBound=lb, upperBound=ub)

        def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
            assert self.intensityproperties is not None, "CTNormalization requires intensity properties"
            # image = image.float()
            mean_intensity = self.intensityproperties['mean']
            std_intensity = self.intensityproperties['std']
            lower_bound = self.intensityproperties['percentile_00_5']
            upper_bound = self.intensityproperties['percentile_99_5']
            # clip_fn = np.clip if type(image) == np.ndarray else torch.clamp
            clip_fn = self.type_to_clip_fn(image)
            image = clip_fn(image, lower_bound, upper_bound)
            image = (image - mean_intensity) / max(std_intensity, 1e-8)
            return image
    
    class ExtractRandomPatchFromPhysicalCoordinates:
        def __init__(self, patch_size_px):
            # (x, y, z)
            self.patch_size_px = patch_size_px
        
        @staticmethod
        def compute_patch_location_mm(x, patch_size_mm):
            # (x, y, z)
            size, spacing = x.GetSize(), x.GetSpacing()
            # max_mm = [s_px * s - p_mm for s_px, s, p_mm in zip(size, spacing, patch_size_mm)]
            orig_mm = x.TransformContinuousIndexToPhysicalPoint((0,0,0))
            max_mm = x.TransformContinuousIndexToPhysicalPoint([i-1 for i in size])
            patch_start_mm = []
            for o, m, p in zip(orig_mm, max_mm, patch_size_mm):
                # if o < m:
                start = np.random.uniform(o, m - p)
                # else:
                #     start = np.random.uniform(o, m + p)
                patch_start_mm.append(start)
            patch_start_mm = np.array(patch_start_mm)
            patch_end_mm = patch_start_mm + patch_size_mm
            return patch_start_mm, patch_end_mm
        
        @staticmethod
        def extract_patch(x, patch_start_mm, patch_end_mm):
            _extract_patch = lambda x, idx: x[idx[0,0]:idx[0,1],idx[1,0]:idx[1,1],idx[2,0]:idx[2,1]]       
            # TODO: is it ensured that these are lower and higher dependent on coordinate system?
            patch_start_px = x.TransformPhysicalPointToIndex(patch_start_mm.tolist())
            patch_end_px = x.TransformPhysicalPointToIndex(patch_end_mm.tolist())
            patch_idx = np.array([[s, e] for s, e in zip(patch_start_px, patch_end_px)])
            patch = _extract_patch(x, patch_idx)
            return patch, patch_idx
        
        def __call__(self, base_img, **child_imgs):
            base_spacing = base_img.GetSpacing()
            # child_imgs = [resample_image(x, out_spacing=base_spacing, interpolator='NearestNeighbor') for x in child_imgs]
            patch_size_mm = [p * s for p, s in zip(self.patch_size_px, base_spacing)]
            patch_start_mm, patch_end_mm = self.compute_patch_location_mm(base_img, patch_size_mm)
            base_patch, base_bbox = self.extract_patch(base_img, patch_start_mm, patch_end_mm)
            # patches = [self.extract_patch(base_img, patch_start_mm, patch_end_mm)]
            patches = {n: self.extract_patch(x, patch_start_mm, patch_end_mm) for n, x in child_imgs.items()}
            patches, bboxes = {n: p[0] for n, p in patches.items()}, {n: p[1] for n, p in patches.items()}
            return base_patch, base_bbox, patches, bboxes

    class Transform:
        def __init__(
            self, 
            intensityproperties, # =INTENSITYPROPERTIES, 
            # target_resolution=None, # (192,192,192),
            target_spacing=None, # (1,1,1)
            patch_size=(96,96,96),
            # num_patches=4,
            radius=4
        ):
            self.ct_normalization = TP.CTNormalization(intensityproperties=intensityproperties)
            # self.target_resolution = target_resolution
            self.target_spacing = target_spacing
            self.patch_size = patch_size
            self.extract_patches = TP.ExtractRandomPatchFromPhysicalCoordinates(patch_size_px=patch_size)
            # self.num_patches = num_patches
            self.radius = radius
        
        @staticmethod
        def pts_to_sitk(pts, radius, ref_sitk):
            pts_np = TP.draw_spheres_from_physical_points(ref_sitk, pts, radius=radius)
            pts_sitk = TP.np_to_sitk(pts_np, ref_sitk)
            return pts_sitk
        
        @staticmethod
        def extract_patch_torch(t, size, bbox):
            t = F.interpolate(t[None], size=size, mode='trilinear', align_corners=True)[0]
            patch = t[:, bbox[0,0]:bbox[0,1],bbox[1,0]:bbox[1,1],bbox[2,0]:bbox[2,1]]
            return patch

        def __call__(self, base_img, y_bilinear={}, y_nearest={}, y_pts={}, y_torch={}, patches=True, target_resolution=None):
            base_img = self.ct_normalization.run(base_img)
            target_spacing = base_img.GetSpacing() if self.target_spacing is None else self.target_spacing
            
            base_img = TP.resample_image(base_img, out_spacing=target_spacing, interpolator='BSpline')
            y_bilinear = {n: TP.resample_image(y, out_spacing=target_spacing, interpolator='BSpline') for n, y in y_bilinear.items()}
            y_nearest = {n: TP.resample_image(y, out_spacing=target_spacing, interpolator='NearestNeighbor') for n, y in y_nearest.items()}

            # if self.target_resolution is not None:
            #     base_img = crop_or_pad_img(base_img, self.target_resolution)
            #     y_bilinear = {n: crop_or_pad_img(y, self.target_resolution) for n, y in y_bilinear.items()}
            #     y_nearest = {n: crop_or_pad_img(y, self.target_resolution) for n, y in y_nearest.items()}
            
            y_pts = {n: self.pts_to_sitk(y.cpu().numpy().reshape(-1,3), self.radius, base_img) for n, y in y_pts.items()}
            if patches:
                child_imgs = {**y_bilinear, **y_nearest, **y_pts}
                base_patch, base_bbox, patches_sitk, bboxes = self.extract_patches(base_img, **child_imgs) #for _ in range(self.num_patches)]
                patches_sitk['x'] = base_patch
                bboxes['x'] = base_bbox
            else:
                if target_resolution is not None:
                    base_img = TP.crop_or_pad_img(base_img, target_resolution)
                y_bilinear = {n: TP.crop_or_pad_img(p, base_img.GetSize()) for n, p in y_bilinear.items()}
                y_nearest = {n: TP.crop_or_pad_seg(p, base_img.GetSize()) for n, p in y_nearest.items()}
                y_pts = {n: TP.crop_or_pad_seg(p, base_img.GetSize()) for n, p in y_pts.items()}
                patches_sitk = {'x': base_img, **y_bilinear, **y_nearest, **y_pts}
                bboxes = None

            patches_torch = {n: TP.sitk_to_torch(p) for n, p in patches_sitk.items()}
            
            if not patches:
                patches_torch = {n: p[None] for n, p in patches_torch.items()}
            # for n, p in patches_torch.items():
            #     if not sum([s==sp for s, sp in zip(p.size()[-3:], self.patch_size)]) == 3:
            #         import pdb;pdb.set_trace()
            else:
                # Calc is trained with different view -> focused
                # Sometimes patch from parent image lies outside boundaries of image
                check_patch = lambda p: p if sum([s==sp for s, sp in zip(p.size()[-3:], self.patch_size)]) == 3 else torch.zeros_like(patches_torch['x'])
                patches_torch = {
                    n:  check_patch(p) for n, p in patches_torch.items()
                }

            # if y_torch is not None:
            for n, y in y_torch.items():
                size = child_imgs[n].GetSize()[::-1]
                bbox = bboxes[n][::-1]
                patches_torch[n] = self.extract_patch_torch(y, size=size, bbox=bbox)

            return patches_torch, patches_sitk, bboxes

    def init_model(self, model_args={'fed_task': 0, 'model_type': 'swin_unetr'}):
        model_type = model_args['model_type']
        if model_type == 'swin_unetr':
            model = self.MultiHeadSwinUNETR(model_args)
        elif model_type == 'nnunet':
            model = self.MultiHeadnnUNet(model_args)
        elif model_type == 'unetr':
            model = self.MultiHeadUNETR(model_args)
        return model

    def init_optimizer(self):
        model = self.model()
        lr = 8e-4
        if type(model) == TP.MultiHeadnnUNet:
            lr = 1e-2
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        return optimizer
    
    def pts_from_pred(self, pred, ref_sitk):
        if type(pred) == torch.Tensor:
            pred = pred.cpu().detach().numpy()
        largest_vols = self.find_largest_volume(pred)
        pts_idx_pred = self.find_center_of_mass(largest_vols)
        pts_pred = np.array([ref_sitk.TransformContinuousIndexToPhysicalPoint(pt_idx.tolist()[::-1]) for pt_idx in pts_idx_pred])
        return pts_idx_pred, pts_pred
    
    @staticmethod
    def sitk_to_torch(img_sitk):
        arr = sitk.GetArrayFromImage(img_sitk)
        # arr = np.einsum('zyx->xyz', arr)
        return torch.from_numpy(arr.astype(np.float32))[None]

    @staticmethod
    def torch_to_sitk(tensor, ref_img, dtype=np.float32):
        img = sitk.GetImageFromArray(tensor.cpu().numpy().astype(dtype))
        img.CopyInformation(ref_img)
        return img
    
    @staticmethod
    def metatensor_to_sitk(tensor):
        t_itk = metatensor_to_itk_image(tensor)
        arr = itk.GetArrayFromImage(t_itk)
        t_sitk = sitk.GetImageFromArray(arr)
        t_sitk.SetSpacing([x for x in t_itk.GetSpacing()])
        t_sitk.SetOrigin([x for x in t_itk.GetOrigin()])
        return t_sitk

    @staticmethod
    def np_to_sitk(arr, ref_sitk):
        img = sitk.GetImageFromArray(arr)
        img.CopyInformation(ref_sitk)
        return img

    @staticmethod
    def draw_spheres_from_physical_points(ref_sitk, points, radius):
        centers = [ref_sitk.TransformPhysicalPointToContinuousIndex(pt) for pt in points]
        volume_size = ref_sitk.GetSize()
        image = sitk.Image(volume_size, sitk.sitkUInt8)
        image_array = sitk.GetArrayFromImage(image)

        for i, center in enumerate(centers):
            ranges = [[int(np.floor(c-radius-1)), int(np.ceil(c+radius+2))] for c in center]
            for x, y, z in itertools.product(range(*ranges[0]), range(*ranges[1]), range(*ranges[2])):
                if ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) <= radius**2:
                    image_array[z, y, x] = i + 1
                        
        # image = sitk.GetImageFromArray(image_array)
        # image.CopyInformation(ref_sitk)
        return image_array

    @staticmethod
    def crop_or_pad_img(img, target_shape=(192,192,192)):
        t = tio.transforms.CropOrPad(target_shape)
        img_tio = tio.ScalarImage.from_sitk(img)
        return t(img_tio).as_sitk()
    
    @staticmethod
    def crop_or_pad_seg(seg, target_shape=(192,192,192)):
        t = tio.transforms.CropOrPad(target_shape)
        seg_tio = tio.LabelMap.from_sitk(seg)
        return t(seg_tio).as_sitk()

    @staticmethod
    def onehot(t, n_cls):
        b, _, h, w, d = t.shape
        t_onehot = torch.zeros((b, n_cls, h, w, d)).to(t.device)
        t_onehot.scatter_(1, t.long(), 1)
        return t_onehot
    
    @staticmethod
    def hard_dice_score(pred_onehot, target):
        tp, fp, fn, _ = get_tp_fp_fn_tn(pred_onehot, target, axes=(0,2,3,4))
        tp = tp.detach().cpu().numpy()
        fp = fp.detach().cpu().numpy()
        fn = fn.detach().cpu().numpy()
        dice = 2 * tp / (2 * tp + fp + fn + 1)
        return dice

    @staticmethod
    def check_patch_sizes(patches_torch, patch_size):
        size_match = True
        for k, v in patches_torch.items():
            if any([s != ps for s, ps in zip(v.size()[-3:], patch_size)]):
            # if not np.all(np.array(v.size()[-3]) == np.array(patch_size)):
                print('SIZE MISMATCH IN', k)
                size_match = False
        return size_match

    @staticmethod
    def append_to_batch(
            patches_torch, patches_sitk,
            batch_torch, batch_sitk
        ):
        for k, v in patches_torch.items():
            if k not in batch_torch:
                batch_torch[k] = []
                batch_sitk[k] = []
            batch_torch[k].append(v)
            batch_sitk[k].append(patches_sitk[k])

    @staticmethod
    def resample_image(
            itk_image,
            new_size=None,
            out_spacing=None,
            interpolator="BSpline"  # "BSpline", "NearestNeighbor" is_label=False
    ) -> sitk.Image:
        # https://www.programcreek.com/python/example/96383/SimpleITK.sitkNearestNeighbor

        if out_spacing is None:
            out_spacing = [
                sz * spc / nsz for nsz, sz, spc in
                zip(new_size, itk_image.GetSize(), itk_image.GetSpacing())
            ]

        original_spacing = itk_image.GetSpacing()
        original_size = itk_image.GetSize()
        
        out_size = [
            int(np.round(original_size[i] * (original_spacing[i] / out_spacing[i]))) for i in range(len(original_size))
        ]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(itk_image.GetDirection())
        resample.SetOutputOrigin(itk_image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

        interpolator = getattr(sitk, f"sitk{interpolator}")
        resample.SetInterpolator(interpolator)

        return resample.Execute(itk_image)
    
    @staticmethod
    def find_largest_volume(seg: np.ndarray) -> np.ndarray:
        seg_c = np.zeros_like(seg)
        for c in np.unique(seg)[1:]:
            labels = cc3d.connected_components(seg==c)
            try:
                # leave out 0 because that is usually largest component
                idx = np.bincount(labels.flatten())[1:].argmax()
                seg_c[labels == idx+1] = c
            except: # Throws error if there is only one component
                pass
        return seg_c.astype(int)

    @staticmethod
    def find_center_of_mass(seg: np.ndarray) -> np.ndarray:
        #seg_labeled = ndimage.label(seg)[0]
        centers = ndimage.center_of_mass(seg, seg, np.unique(seg)[1:])
        return np.array(centers)
    
    @staticmethod
    def mps_from_pts(pts, path):
        xml_from_pt = lambda i, pt: f"""<point>
                    <id>{i}</id>
                    <specification>0</specification>
                    <x>{pt[0]}</x>
                    <y>{pt[1]}</y>
                    <z>{pt[2]}</z>
                </point>
                """
        mps_string = f"""<?xml version="1.0" encoding="UTF-8"?>
<point_set_file>
    <file_version>0.1</file_version>
    <point_set>
        <time_series>
            <time_series_id>0</time_series_id>
            <Geometry3D ImageGeometry="false" FrameOfReferenceID="0">
                <IndexToWorld type="Matrix3x3" m_0_0="1" m_0_1="0" m_0_2="0" m_1_0="0" m_1_1="1" m_1_2="0" m_2_0="0" m_2_1="0" m_2_2="1"/>
                <Offset type="Vector3D" x="0" y="0" z="0"/>
                <Bounds>
                    <Min type="Vector3D" x="{np.min(pts[:,0])}" y="{np.min(pts[:,1])}" z="{np.min(pts[:,2])}"/>
                    <Max type="Vector3D" x="{np.max(pts[:,0])}" y="{np.max(pts[:,1])}" z="{np.max(pts[:,2])}"/>
                </Bounds>
            </Geometry3D>
            {''.join([xml_from_pt(i, pt) for i, pt in enumerate(pts)])}
        </time_series>
    </point_set>
</point_set_file>"""
        with open(path, 'w') as f:
            f.write(mps_string)
    
    def get_e_metrics(
        self,
        student_logits,
        batch_torch,
        y_pts,
        # e_metrics,
        base_img,
        # mode='val'
    ):  
        e_metrics = {}
        y_pts = {n: y.cpu().numpy().reshape(-1,3) for n, y in y_pts.items()}
        for k, l in student_logits.items(): # zip(student_logits, ['seg', 'hps' , 'ms', 'calc']):
            if k not in batch_torch:
                continue

            l = l.detach().cpu()
            labels = self.key_to_labels[k]
            y = batch_torch[k]
            pred = torch.argmax(torch.softmax(l.detach(), dim=1), dim=1, keepdim=True).cpu()
            pred_onehot = self.onehot(pred, n_cls=len(labels)+1)[:,1:]
            y_onehot = self.onehot(y, n_cls=len(labels)+1)[:,1:]
            acc = (pred == y).sum() / np.prod(y.shape)
            dice = self.hard_dice_score(pred_onehot, y_onehot)
            dice = np.nan_to_num(dice)
            
            for d, l in zip(dice, labels):
                _k = f'{l}_DICE'
                e_metrics[_k] = d

            dice = dice.mean()
            _k = f'{k}_DICE'
            e_metrics[_k] = dice
            _k = f'{k}_acc'
            e_metrics[_k] = acc.item()

            # pred_sitk = torch_to_sitk(pred[0,0], base_img)
            # sitk.WriteImage(pred_sitk, subject / f'kd_{k}.nii.gz')

            # sitk.WriteImage(base_img, '/data/base_img.nii.gz')
            # sitk.WriteImage(self.torch_to_sitk(pred[0,0], base_img), '/data/hps.nii.gz')

            if k in ['hps', 'ms']:
                pts_idx_pred, pts_pred = self.pts_from_pred(pred[0,0], base_img)
                pts_gt = y_pts[k]
                idx_preds = torch.unique(pred)[1:]
                i = 0
                for j, pt_label in enumerate(labels):
                    k = f'{pt_label}_mm'
                    if j+1 in idx_preds:
                        pt_pred = pts_pred[i]
                        pt_gt = pts_gt[i]
                        i += 1
                        d = np.linalg.norm(pt_pred - pt_gt)
                    else:
                        d = np.nan
                    e_metrics[k] = d
        return e_metrics

    def get_ys(self, y, fed_task):
        if fed_task == 0:
            y_nearest = {n: y[k] for k, n in self.Y_NEAREST_KEYS.items()}
            y_nearest = {n: self.metatensor_to_sitk(p) for n, p in y_nearest.items()}
            y_pts = {n: y[k].cpu() for k, n in self.Y_PTS_KEYS.items()}
        elif fed_task == 1:
            y_nearest = {'heart': self.metatensor_to_sitk(y['Heart Seg'])}
            l, n = self.fed_task_to_y_key[fed_task]
            y_pts = {n: y[l]}
        elif fed_task == 2:
            y_nearest = {'heart': self.metatensor_to_sitk(y['Heart Seg'])}
            l, n = self.fed_task_to_y_key[fed_task]
            y_pts = {n: y[l]}
        elif fed_task == 3:
            y_pts = {}
            l, n = self.fed_task_to_y_key[fed_task]
            y_nearest = {n: self.metatensor_to_sitk(y[l]), 'heart': self.metatensor_to_sitk(y['Heart Seg'])}
        return y_nearest, y_pts

    def training_data(self, batch_size=1):
        model = self.model()
        fed_task = int(model.fed_task.item())

        if fed_task == 0:
            y_keys = list(self.Y_NEAREST_KEYS) + list(self.Y_PTS_KEYS)
        else:
            self.X_KEY = self.X_KEY.replace(' New Origin', '')
            y_keys = [self.fed_task_to_y_key[fed_task][0]]
            y_keys.append('Heart Seg')
        
        dataset = MedicalFolderDataset(
            root=self.dataset_path,
            data_modalities=[self.X_KEY],
            target_modalities=y_keys,
            transform=None,
            target_transform=None,
            demographics_transform=None
        )
        loader_arguments = {'batch_size': batch_size, 'shuffle': True}
        return DataManager(dataset, **loader_arguments)

    def training_step(self, data, y):
        try:
            model = self.model()
            fed_task = int(model.fed_task.item())
            condition_on_seg = model.condition_on_seg.item()
            output_seg = model.output_seg.item()
            patches = model.patches.item()
            deep_supervision = model.deep_supervision.item()

            
            base_img = data[0][self.X_KEY]
            base_img = self.metatensor_to_sitk(base_img)
            
            y_nearest, y_pts = self.get_ys(y, fed_task)

            if patches:
                batch_torch, batch_sitk = {}, {}
                i = 0
                
                while i < 4:
                    
                    patches_torch, patches_sitk, bboxes = self.t(
                        base_img=base_img,
                        y_nearest=y_nearest,
                        y_pts=y_pts
                    )

                    if not self.check_patch_sizes(patches_torch, self.patch_size):
                        return torch.tensor([0.])
                        
                    i += 1
                    
                    self.append_to_batch(
                        patches_torch, patches_sitk,
                        batch_torch, batch_sitk
                    )
            else:
                batch_torch, batch_sitk, bboxes = self.t(
                    base_img=base_img,
                    y_nearest=y_nearest,
                    y_pts=y_pts,
                    patches=False,
                    target_resolution=self.target_resolution
                )

            if patches:
                batch_torch = {k: torch.stack(v) for k, v in batch_torch.items()}
            x = batch_torch['x']

            if condition_on_seg:
                x = torch.cat([x, batch_torch['heart']], dim=1)
            
            student_logits = model(x.to(self._device))

            loss = torch.tensor(0.).to(self._device)
            
            for k, logits in student_logits.items():
                if k in batch_torch:
                    y = batch_torch[k]
                    if deep_supervision:
                        shapes = [o.shape[2:] for o in logits[:3]]
                        ys = [F.interpolate(y, size=sh, mode='nearest') for sh in shapes]
                        for f, l, t in zip(self.weight_factors[:3], logits[:3], ys[:3]):
                            loss += f * self.criterion(l, t.to(self._device))
                    else:
                        loss += self.criterion(logits, batch_torch[k].to(self._device))
            
            # torch.save(model.state_dict(), f'/data/{model.__class__.__name__}_{fed_task}_hps.pt')
        except:
            loss = torch.tensor(0.).to(self._device)
        
        # tmp_dir = Path('/data/tmp')
        # tmp_dir.mkdir(exist_ok=True)
        # for n, p in patches_sitk.items():
        #     sitk.WriteImage(p, tmp_dir / f'{n}.nii.gz')

        # base_img = patches_sitk['x']
        # for k, v in student_logits.items():
        #     pred = torch.softmax(v, dim=1).argmax(dim=1)[0].cpu().detach()
        #     pred = self.torch_to_sitk(pred, base_img)
        #     sitk.WriteImage(pred, tmp_dir / f'p_{k}.nii.gz')
        # for k, v in patches_torch.items():
        #     # pred = torch.softmax(v, dim=1).argmax(dim=1)[0].cpu().detach()
        #     pred = self.torch_to_sitk(v[0], base_img)
        #     sitk.WriteImage(pred, tmp_dir / f't_{k}.nii.gz')
        # import pdb;pdb.set_trace()

        torch.cuda.empty_cache()
        return loss

    def testing_step(self, data, y):
        # try:
        model = self.model().cuda()
        fed_task = int(model.fed_task.item())
        condition_on_seg = model.condition_on_seg.item()
        output_seg = model.output_seg.item()
        patches = model.patches.item()
        deep_supervision = model.deep_supervision.item()

        base_img = data[0][self.X_KEY]
        base_img = self.metatensor_to_sitk(base_img)
        
        y_nearest, y_pts = self.get_ys(y, fed_task)
        # img_fname = Path(img.meta['filename_or_obj'])
        # series_dir = img_fname.parents[1]
        
        target_resolution = self.target_resolution if not patches else None
        batch_torch, batch_sitk, bboxes = self.t(
            base_img=base_img,
            y_nearest=y_nearest,
            y_pts=y_pts,
            patches=False,
            target_resolution=target_resolution
        )
        
        x = batch_torch['x'] 
        if condition_on_seg:
            x = torch.cat([x, batch_torch['heart']], dim=1)
        
        if patches:
            with torch.cuda.amp.autocast():
                student_logits = sliding_window_inference(x.cuda(), self.patch_size, 4, model)
        else:
            student_logits = model(x.cuda())
        
        model.cpu()
        
        if deep_supervision:
            student_logits = {k: s[0] for k, s in student_logits.items()}
        student_logits = {k: s.cpu() for k, s in student_logits.items()}
        
        loss = torch.tensor(0.)
        for k, logits in student_logits.items():
            if k in batch_torch:
                y = batch_torch[k]
                # if deep_supervision:
                #     shapes = [o.shape[2:] for o in logits[:3]]
                #     ys = [F.interpolate(y, size=sh, mode='nearest') for sh in shapes]
                #     for f, l, t in zip(self.weight_factors[:3], logits[:3], ys[:3]):
                #         loss += f * self.criterion(l, t.to(self._device))
                #     pass
                # else:
                loss += self.criterion(logits, batch_torch[k])

        e_metrics = self.get_e_metrics(
            student_logits=student_logits,
            batch_torch=batch_torch,
            y_pts=y_pts,
            base_img=batch_sitk['x'],
        )
        e_metrics['Loss'] = loss.item()
        
        torch.cuda.empty_cache()
        return e_metrics
        # except:
        #     return {'Loss': 0.}

if __name__ == '__main__':
    from argparse import ArgumentParser
    from federated import federated_experiment
    parser = ArgumentParser()
    parser.add_argument('--task', default=0)
    parser.add_argument('--tags', '-t', nargs='+')
    # parser.add_argument('--config', '-c')
    parser.add_argument('--locations', '-l', nargs='+', default=None)
    parser.add_argument('--mode', '-m', default='train')
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--batch_maxnum', type=int, default=None)
    parser.add_argument('--test_on_global_updates', action='store_true', default=False)
    parser.add_argument('--test_on_local_updates', action='store_true', default=False)
    parser.add_argument('--exp_name', '-e', default='fed-tavi-pts-detection')
    parser.add_argument('--local', action='store_true', default=False)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--ckpt')
    parser.add_argument('--condition_on_seg', action='store_true', default=False)
    parser.add_argument('--output_seg', action='store_true', default=False)
    parser.add_argument('--patches', action='store_true', default=False)
    parser.add_argument('--model_type', default='swin_unetr') # nnunet, unetr
    args = parser.parse_args()

    # os.environ['FED_TASK'] = args.task
    # num_segmentation_heads = len(UNetTrainingPlan.y_key_to_seg_labels[UNetTrainingPlan.fed_task_to_y_key[int(args.task)]])+1
    # model_args = {'num_segmentation_heads': num_segmentation_heads, 'fed_task': args.task}
    # print(model_args)
    in_channels = 2 if args.condition_on_seg else 1
    out_channels = {'hps': 6, 'ms': 3, 'calc': 2}
    if args.output_seg:
        out_channels['heart'] = 8
    img_size = (96,96,96) if args.patches else (192,192,192)

    model_args = {
        'fed_task': args.task, 
        'finetune': args.finetune,
        'in_channels': in_channels,
        'out_channels': out_channels,
        'img_size': img_size,
        'model_type': args.model_type,
        'patches': float(args.patches),
        'deep_supervision': 1. if args.model_type == 'nnunet' else 0.,
        'condition_on_seg': float(args.condition_on_seg),
        'output_seg': float(args.output_seg)
    }
    federated_experiment(training_plan=TP, args=args, model_args=model_args)