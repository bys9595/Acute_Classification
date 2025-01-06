import torch
import torch.nn.functional as F
from monai.networks.nets import ViT


class VisionTransformer(ViT):
    def __init__(
            self,
            in_channels=1,
            img_size=(96, 96, 96),
            patch_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            pos_embed_type='sincos',
            classification=True,
            num_classes=2,
            spatial_dims=3,
            post_activation=None,
            qkv_bias=True,
            global_pool='avg'
        ):
        super().__init__(in_channels=in_channels, img_size=img_size, patch_size=patch_size, hidden_size=hidden_size, mlp_dim=mlp_dim,
                         num_layers=num_layers, num_heads=num_heads, pos_embed_type=pos_embed_type, classification=classification,
                         num_classes=num_classes, spatial_dims=spatial_dims, post_activation=post_activation, qkv_bias=qkv_bias)
        self.global_pool = global_pool
        
    def forward(self, x):
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        
        if self.global_pool == 'avg':
            x = x[:, 1:].mean(dim=1)
        elif self.global_pool == 'token':
            x = x[:, 0]  # class token
    
        if hasattr(self, "classification_head"):
            x = self.classification_head(x)
            
        return x



if __name__ == "__main__":
    
    img = torch.randn((1, 1, 128, 128, 64)).cuda()
    model = VisionTransformer(in_channels=1,
            img_size=(128, 128, 64),
            num_classes=1,
            global_pool='avg').cuda()
    
    out = model(img)
    
    a = 1
    
    