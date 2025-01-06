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

import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather
import torch.nn.functional as F
from monai.data import decollate_batch
import logging
from utils.metrics import BinaryEvalMetrics
import monai
import random
import torch.distributed as dist
import gc

def train_epoch(model, loader, optimizer, scaler, epoch, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()

    loss_func = torch.nn.BCEWithLogitsLoss()
    train_metrics = BinaryEvalMetrics()

    optimizer.zero_grad()  # 에포크 시작 시 그래디언트 초기화

    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        
        target = target.float()

        data, target = data.cuda(args.rank), target.cuda(args.rank)
        
        with autocast(enabled=args.amp):
            logits = model(data)
            if isinstance(logits, monai.data.meta_tensor.MetaTensor):
                logits = logits.as_tensor()
            loss = loss_func(logits.squeeze(1), target)
            # 그래디언트 누적을 위해 손실을 누적 단계 수로 나눕니다
            loss = loss / args.gradient_accumulation_steps

        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
            
        # 그래디언트 누적 스텝이 완료되었을 때만 동기화 및 옵티마이저 스텝 수행
        if (idx + 1) % args.gradient_accumulation_steps == 0 or (idx + 1) == len(loader):  # 수정된 부분
            # 분산 학습일 때만 그래디언트 동기화 수행
            if args.distributed:
                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= args.world_size
            
            if args.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            
        run_loss.update(loss.item() * args.gradient_accumulation_steps, n=args.batch_size)
        train_metrics.update(logits, target)

        gc.collect()
 
        if args.rank == 0 and (idx + 1) % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            logging.info(
                "[Train] Epoch {}/{} {}/{} loss: {:.4f} lr: {:.6f} time {:.2f}s".format(
                    epoch, args.max_epochs, idx+1, len(loader),
                    run_loss.avg, lr, time.time() - start_time
                )
            )
            start_time = time.time()
    
    if args.distributed:
        torch.distributed.barrier()
    
    auc = train_metrics.compute()
    
    return run_loss.avg, auc.item()


def val_epoch(model, loader, epoch, args):
    model.eval()
    start_time = time.time()
    run_loss = AverageMeter()
    
    loss_func = torch.nn.BCEWithLogitsLoss()
    val_metrics = BinaryEvalMetrics()
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]

            target = target.float()
            
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            
            with autocast(enabled=args.amp):
                logits = model(data)
                if isinstance(logits, monai.data.meta_tensor.MetaTensor):
                    logits = logits.as_tensor()
                loss = loss_func(logits.squeeze(1), target)
            
            run_loss.update(loss.item(), n=args.batch_size)
            val_metrics.update(logits, target)
           
            if args.rank == 0 and (idx + 1) % args.print_freq == 0:
                print(
                "[Validation]"
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx+1, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
                start_time = time.time()
    
    if args.distributed:
        torch.distributed.barrier()
    
    auc = val_metrics.compute()
    
    return run_loss.avg, auc.item()


def save_checkpoint(model, epoch, args, filename="model.pt", best_auc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {
        "epoch": epoch,
        "best_auc": best_auc,
        "state_dict": state_dict,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all()  # 모든 GPU의 상태 저장
        }
    }
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)

def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    args,
    scheduler=None,
    start_epoch=0,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    
    # auc가 아니라 multiclass인 경우에는 accuracy이기 때문에 변수 이름을 적절히 변경
    val_auc_max = 0.0
    
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        epoch_time = time.time()
        train_loss, train_auc = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, args=args
        )
        torch.cuda.empty_cache()
        gc.collect()
        if args.rank == 0:
            logging.info("Final training  {}/{}   loss: {:.4f}   auc: {:.4f}   time {:.2f}s".format(epoch, args.max_epochs - 1, train_loss, train_auc, time.time() - epoch_time))

        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_auc", train_auc, epoch)
        
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()

            val_avg_loss, val_avg_auc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                args=args,
            )

            if args.rank == 0:
                logging.info("Final validation  {}/{}   auc {}   time {:.2f}s".format(epoch, args.max_epochs - 1, val_avg_auc, time.time() - epoch_time))

                if writer is not None:
                    writer.add_scalar("val_auc", val_avg_auc, epoch)
                if val_avg_auc > val_auc_max:
                    logging.info("new best auc / accuracy ({:.6f} --> {:.6f}). ".format(val_auc_max, val_avg_auc))
                    val_auc_max = val_avg_auc
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_auc=val_auc_max, filename="model_best_auc.pt",  optimizer=optimizer, scheduler=scheduler
                        )

        if scheduler is not None:
            scheduler.step()

    if args.rank == 0:
        logging.info("Training Finished !, Best AUC/Accuracy: {:.6f}".format(val_auc_max))

    return val_auc_max