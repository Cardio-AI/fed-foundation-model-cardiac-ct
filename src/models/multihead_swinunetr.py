import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.networks.blocks import UnetOutBlock

class MultiHeadSwinUNETR(nn.Module):
    def __init__(
            self, 
            out_channels,
            img_size=(96,96,96), 
            in_channels=1, 
            intermediate_out_channels=1,
            ckpt_path=None
        ):
        super().__init__()
        spatial_dims = 3
        feature_size = 48
        self.swin_unetr = SwinUNETR(
            img_size=img_size, # (96,96,96),
            in_channels=in_channels, # 1,
            out_channels=intermediate_out_channels, # 1,
            feature_size=feature_size,
            use_checkpoint=True,    
        ).cuda()
        if ckpt_path is not None:
            self.swin_unetr.load_from(
                weights=torch.load(ckpt_path) # "./checkpoints/model_swinvit.pt")
            )

        block = lambda oc: UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=oc)
        self.outs = nn.ModuleDict({n: block(oc) for n, oc in out_channels.items()})

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