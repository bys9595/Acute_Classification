# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer_multiclass import run_training
from utils.data_utils import get_loader_multiclass

from timm.utils import setup_default_logging
from models.create_model import create_model
import torch.backends.cudnn as cudnn
import random
import timm.optim.optim_factory as optim_factory

# 학습 코드 완료 후 파일 마지막에 추가
import gc
import torch
import os


import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
print('Setting resource limit:', str(resource.getrlimit(resource.RLIMIT_NOFILE)))


parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="logs", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--pretrained_checkpoint",default="", type=str, help="")
parser.add_argument("--save_checkpoint", default=True, help="save checkpoint during training")

parser.add_argument("--task_name", default="4Class+Normal", type=str, help="task name")
parser.add_argument("--years", default=['2014', '2015', '2016', '2018', '2019', '2020_1', '2020_3',
                                        '2021_1', '2021_2', '2021_3', '2021_4', '2021_6', '2022_2023'], type=list, help="years")
parser.add_argument("--model", default="swint", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--batch_size", default=3, type=int, help="number of batch size")
parser.add_argument("--optim_lr", default=5e-5, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--layer_decay", default=0.75, type=float, help="layer-wise learning rate decay")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:34567", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=5, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=275, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--roi_x", default=192, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=192, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=64, type=int, help="roi size in z direction")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.5, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
# warmup is important !!!
parser.add_argument("--max_epochs", default=50, type=int, help="max number of training epochs")
parser.add_argument("--warmup_epochs", default=5, type=int, help="number of warmup epochs")
parser.add_argument("--early_stop_epoch", default=10, type=int, help="random seed for reproducibility")
parser.add_argument("--val_every", default=1, type=int, help="validation frequency")
parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
parser.add_argument("--use_normal_dataset", default=True, help="use monai Dataset class")
parser.add_argument("--gradient_accumulation_steps", default=10, type=int, help="number of gradient accumulation steps")


parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")
parser.add_argument("--use_ssl_pretrained", default=False, help="use self-supervised pretrained weights")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")

parser.add_argument("--seed", default=0, type=int, help="random seed for reproducibility")
parser.add_argument('--fold_num', default='0', type=str, choices=['0', '1', '2', '3', '4'])
parser.add_argument('--load_dict_name', default='student', type=str, choices=['student', 'teacher'])
parser.add_argument("--majority_undersampling", action="store_true", help="majority undersampling")

def seed_everything(args, rng_state=None)->None:
    # 재현성을 위해 시드 고정
    seed = args.seed
    deterministic = True

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    
    if rng_state is None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        random.setstate(rng_state['python'])
        np.random.set_state(rng_state['numpy'])
        torch.set_rng_state(rng_state['torch'])
        torch.cuda.set_rng_state(rng_state['torch_cuda'])
    
    os.environ["PL_SEED_WORKERS"] = f"{int(args.num_workers)}"
    
    cudnn.benchmark = (not deterministic)
    cudnn.deterministic = deterministic


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)
    
    # CUDA 캐시 정리
    torch.cuda.empty_cache()

    # Garbage collection 강제 실행
    gc.collect()


def change_weight_key(state_dict, target_key='module.'):
    print("Tag '", target_key, "' found in state dict - fixing!")
    for key in list(state_dict.keys()):
        if target_key in key:
            state_dict[key.replace(target_key, "")] = state_dict.pop(key)
        
import datetime
def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("spawn", force=True)
        torch.multiprocessing.set_sharing_strategy('file_descriptor')
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
            timeout=datetime.timedelta(minutes=30)
        )
    torch.cuda.set_device(args.gpu)
    
    torch.backends.cudnn.enabled = True
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        if "rng_state" in checkpoint:
            seed_everything(args, checkpoint["rng_state"])
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))
    else:
        seed_everything(args)
    
    print(args.rank, " gpu", args.gpu)
    
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)

    
    if args.rank == 0:
        os.makedirs(args.logdir, exist_ok=True)
        setup_default_logging(log_path=os.path.join(args.logdir, 'train.txt'))
    logger = init_log('global', logging.INFO)
    logger.propagate = False
    
    model = create_model(args)

    if args.use_ssl_pretrained:
        try:            
            model_dict = torch.load(args.pretrained_checkpoint, map_location=torch.device('cpu'))
            
            if 'GBT' in args.pretrained_checkpoint and args.model in ['swint', 'swint_v2']:
                state_dict = model_dict[args.load_dict_name]
                change_weight_key(state_dict, 'module.')
                
                for key in list(state_dict.keys()):
                    if 'swinViT.' not in key:
                        state_dict.pop(key)
                change_weight_key(state_dict, 'swinViT.')
                
            else:
                raise ValueError("Self-supervised pre-trained weights not available, " + str(args.pretrained_checkpoint))
                
                
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print("Using pretrained self-supervised weights !")
            
            if args.rank == 0:
                # 변경된 키들을 출력합니다.
                print("missing_keys:", missing_keys)
                print("unexpected_keys:", unexpected_keys)
            
        except ValueError:
            raise ValueError("Self-supervised pre-trained weights not available for" + str(args.model_name))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    loader = get_loader_multiclass(args)
    
    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)
    model_without_ddp = model
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model_without_ddp = model
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu],
            output_device=args.gpu,
            find_unused_parameters=False,
            broadcast_buffers=False
        )
    
    from utils.optim_factory import LayerDecayValueAssigner, create_optimizer
    if args.use_ssl_pretrained:
        num_layers = sum(model_without_ddp.depths)

        if args.layer_decay < 1.0 and 'swin' in args.model:
            assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)),
                                            is_swin=True,
                                            depths=model_without_ddp.depths)
        elif args.layer_decay < 1.0:
            assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        else:
            assigner = None
    else:
        assigner = None
    
    optimizer = create_optimizer(
            args, model_without_ddp,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)


    
    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs, warmup_start_lr=1e-6
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    
    
    import json
    if args.logdir and args.rank == 0:
        with open(os.path.join(args.logdir, 'config.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        args=args,
        scheduler=scheduler,
        start_epoch=start_epoch,
    )
    torch.cuda.empty_cache()  # 주기적으로 캐시 정리
    return accuracy

logs = set()
def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


if __name__ == "__main__":
    main()
