import torch.nn.functional as F
from monai.networks import nets
from monai.networks.nets import ViT
        
class UNETR(nets.UNETR):
    def __init__(
            self,
            in_channels=1,
            out_channels=1,
            img_size=(96, 96, 96),
            feature_size=64,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed_type='sincos',
            qkv_bias=True
        ):
        super().__init__(in_channels=in_channels, out_channels=out_channels, img_size=img_size, feature_size=feature_size, 
                         hidden_size=hidden_size, mlp_dim=mlp_dim, num_heads=num_heads, qkv_bias=qkv_bias)

        # Transformer Encoder     
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=16,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=12,
            num_heads=num_heads,
            pos_embed_type=pos_embed_type,
            classification=False,
            spatial_dims=3,
            qkv_bias=qkv_bias,
            )


if __name__ == "__main__":
    net = UNETR()
    
    print(net)