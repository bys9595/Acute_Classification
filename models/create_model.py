from models.vision_transformer import VisionTransformer
from models.swin_transformer import SwinTransformer as SwinT_OLD
from models.swin_transformer_new import SwinTransformer, SwinTransformer_v2

def create_model(args):        
    if args.model == 'vit':
        model = VisionTransformer(
            in_channels=args.in_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            num_classes=args.out_channels,
            global_pool='avg'
        )
        
    elif args.model == 'swint_old':
        model = SwinT_OLD(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            patch_size=2,
            in_chans=args.in_channels,
            num_classes=args.out_channels,
            embed_dim=args.feature_size,
            window_size=7,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            drop_rate=0.0,
            attn_drop_rate=0.0,
            use_checkpoint=args.use_checkpoint,
        )
    
    elif args.model == 'swint':
        model = SwinTransformer(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            patch_size=2,
            in_chans=args.in_channels,
            num_classes=args.out_channels,
            embed_dim=args.feature_size,
            window_size=7,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            drop_rate=0.0,
            attn_drop_rate=0.0,
            use_checkpoint=args.use_checkpoint,
        )   
        
    elif args.model == 'swint_v2':
        model = SwinTransformer_v2(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            patch_size=2,
            in_chans=args.in_channels,
            num_classes=args.out_channels,
            embed_dim=args.feature_size,
            window_size=7,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            drop_rate=0.0,
            attn_drop_rate=0.0,
            use_checkpoint=args.use_checkpoint,
        )
        
    elif args.model == 'swint_v2_moe':
        model = SwinTransformer_v2(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            patch_size=2,
            in_chans=args.in_channels,
            num_classes=args.out_channels,
            embed_dim=args.feature_size,
            window_size=7,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            drop_rate=0.0,
            attn_drop_rate=0.0,
            use_checkpoint=args.use_checkpoint,
            moe=True,
        )

    else:
        ValueError('Please check the model name')

    return model