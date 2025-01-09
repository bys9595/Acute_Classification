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

import math
import os
import pickle
import numpy as np
import torch
import torch.distributed as dist

import itertools as it
from monai import data, transforms
from monai.data import *
import logging
import warnings

# nibabel의 로그 레벨을 ERROR로 설정하여 경고를 무시
logging.getLogger('nibabel').setLevel(logging.ERROR)

# 특정 경고 메시지를 무시하도록 설정
warnings.filterwarnings("ignore", message="pixdim[0] (qfac) should be 1 (default) or -1; setting qfac to 1")

class BalancedSampler(torch.utils.data.Sampler):
    def __init__(
        self, 
        dataset, 
        labels,  
        num_replicas=None, 
        rank=None, 
        shuffle=True, 
        sampling_strategy='under',
        distributed=False,
        make_even=True  # make_even 옵션 추가
    ):
        self.distributed = distributed
        
        if distributed:
            if num_replicas is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                num_replicas = dist.get_world_size()
            if rank is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                rank = dist.get_rank()
        else:
            num_replicas = 1
            rank = 0
            
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.labels = labels
        self.sampling_strategy = sampling_strategy
        self.make_even = make_even
        
        # 클래스별 인덱스 저장
        label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
            
        # 클래스별 샘플 수 계산
        class_counts = {label: len(indices) for label, indices in label_to_indices.items()}
        
        # sampling strategy에 따라 각 클래스별 샘플 수 조정
        if sampling_strategy == 'under':
            min_count = min(class_counts.values())
            self.num_samples_per_class = min_count
        else:  # over sampling
            max_count = max(class_counts.values())
            self.num_samples_per_class = max_count
            
        # 총 샘플 수 계산
        total_samples = self.num_samples_per_class * len(class_counts)
        
        if distributed:
            self.total_size = int(math.ceil(total_samples / self.num_replicas)) * self.num_replicas
            self.num_samples = self.total_size // self.num_replicas
        else:
            # 단일 GPU에서는 전체 샘플 수 사용
            self.num_samples = total_samples
            self.total_size = total_samples
        
        self.label_to_indices = label_to_indices
        self.class_counts = class_counts

        # 원본 클래스별 샘플 수 출력
        if self.rank == 0:
            print("\n=== Original class distribution ===")
            total_original = 0
            for label, count in class_counts.items():
                print(f"Class {label}: {count} samples")
                total_original += count
            print(f"Total samples: {total_original}")
            
            # sampling strategy에 따른 조정 후 클래스별 샘플 수 출력
            print(f"\n=== After {sampling_strategy}-sampling ===")
            total_balanced = 0
            for label in class_counts.keys():
                balanced_count = self.num_samples_per_class
                print(f"Class {label}: {balanced_count} samples")
                total_balanced += balanced_count
            print(f"Total samples: {total_balanced}")
            
            if distributed:
                samples_per_gpu = int(math.ceil(total_balanced * 1.0 / self.num_replicas))
                print(f"\nDistributed mode: {samples_per_gpu} samples per GPU")
    
    def __iter__(self):
        indices = []
        for label, label_indices in self.label_to_indices.items():
            if self.sampling_strategy == 'under':
                if len(label_indices) > self.num_samples_per_class:
                    rand_idx = torch.randperm(len(label_indices))[:self.num_samples_per_class]
                    selected = torch.tensor(label_indices)[rand_idx].tolist()
                else:
                    selected = label_indices
            else:
                selected = torch.randint(0, len(label_indices), (self.num_samples_per_class,)).tolist()
            indices.extend(selected)

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.tensor(indices)[torch.randperm(len(indices), generator=g)].tolist()

        if self.distributed and self.make_even:
            extra_indices = indices[:(self.total_size - len(indices))]
            indices.extend(extra_indices)
            assert len(indices) == self.total_size

        if self.distributed:
            indices = indices[self.rank:self.total_size:self.num_replicas]

        return iter(indices)

    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        
        
        
class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(args):
    task_name = args.task_name
    fold_num = args.fold_num
    years = args.years
    
    # 데이터셋 경로 설정
    datadir_list = []
    jsonlist = []
    for year in years:
        datadir_list.append("/mai_nas/Private_Dataset/GangNam_SEV/" + year + "/")
        jsonlist.append("/mai_nas/Private_Dataset/GangNam_SEV/jsons/" + year + "/" 
                        + task_name + "/dataset_GNSEV_" + year + "_fold" + fold_num + ".json")
    
    # datalist 생성
    datalist_train = []
    datalist_val = []
    for json_path, base_dir in zip(jsonlist, datadir_list):
        # Training Data List
        datalist_i = load_decathlon_datalist(json_path, False, "training", base_dir=base_dir)
        for item in datalist_i:
            datalist_train.append({"image": item["image"], "label": item["label"]})
        
        # Validation Data List
        datalist_val_i = load_decathlon_datalist(json_path, False, "validation", base_dir=base_dir)
        for item in datalist_val_i:
            datalist_val.append({"image": item["image"], "label": item["label"]})

    print("Dataset all training: number of data: {}".format(len(datalist_train)))
    print("Dataset all validation: number of data: {}".format(len(datalist_val)))

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"], image_only=True),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.CropForegroundd(keys=["image"], source_key="image", allow_smaller=False),
            transforms.Resized(keys=["image"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.RandFlipd(keys=["image"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image"]) if args.out_channels == 1 else transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"], image_only=True),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.CropForegroundd(keys=["image"], source_key="image", allow_smaller=False),
            transforms.Resized(keys=["image"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.ToTensord(keys=["image"]) if args.out_channels == 1 else transforms.ToTensord(keys=["image", "label"]),
        ]
    )        

    if args.use_normal_dataset:
        train_ds = data.Dataset(data=datalist_train, transform=train_transform)
    else:
        train_ds = data.SmartCacheDataset(data=datalist_train, transform=train_transform, replace_rate=1.0, cache_num=int(2*args.batch_size))
    
    if args.distributed:
        if args.majority_undersampling:
            train_sampler = BalancedSampler(train_ds, labels=[item["label"] for item in datalist_train],
                                            sampling_strategy='under', distributed=True)
        else:
            train_sampler = Sampler(train_ds)
    else:
        if args.majority_undersampling:
            train_sampler = BalancedSampler(train_ds, labels=[item["label"] for item in datalist_train],
                                            sampling_strategy='under', distributed=False)
        else:
            train_sampler = None
    
    train_loader = data.ThreadDataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler
        )

    # Validation Dataloader
    val_ds = data.Dataset(data=datalist_val, transform=val_transform)
            
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None

    val_loader = data.ThreadDataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=val_sampler)
    loader = [train_loader, val_loader]
    
    return loader


def get_loader_multiclass(args):
    task_name = args.task_name
    fold_num = args.fold_num
    years = args.years
    
    # 데이터셋 경로 설정
    datadir_list = []
    jsonlist = []
    for year in years:
        datadir_list.append("/mai_nas/Private_Dataset/GangNam_SEV/" + year + "/")
        jsonlist.append("/mai_nas/Private_Dataset/GangNam_SEV/jsons_multiclass/" + year + "/" 
                        + task_name + "/dataset_GNSEV_" + year + "_fold" + fold_num + ".json")
    
    # datalist 생성
    datalist_train = []
    datalist_val = []
    for json_path, base_dir in zip(jsonlist, datadir_list):
        # Training Data List
        datalist_i = load_decathlon_datalist(json_path, False, "training", base_dir=base_dir)
        for item in datalist_i:
            datalist_train.append({"image": item["image"], "label": item["label"]})
        
        # Validation Data List
        datalist_val_i = load_decathlon_datalist(json_path, False, "validation", base_dir=base_dir)
        for item in datalist_val_i:
            datalist_val.append({"image": item["image"], "label": item["label"]})

    print("Dataset all training: number of data: {}".format(len(datalist_train)))
    print("Dataset all validation: number of data: {}".format(len(datalist_val)))

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"], image_only=True),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.CropForegroundd(keys=["image"], source_key="image", allow_smaller=False),
            transforms.Resized(keys=["image"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.RandFlipd(keys=["image"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image"]) if args.out_channels == 1 else transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"], image_only=True),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.CropForegroundd(keys=["image"], source_key="image", allow_smaller=False),
            transforms.Resized(keys=["image"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.ToTensord(keys=["image"]) if args.out_channels == 1 else transforms.ToTensord(keys=["image", "label"]),
        ]
    )        

    if args.use_normal_dataset:
        train_ds = data.Dataset(data=datalist_train, transform=train_transform)
    else:
        train_ds = data.SmartCacheDataset(data=datalist_train, transform=train_transform, replace_rate=1.0, cache_num=int(2*args.batch_size))
        
    if args.distributed:
        if args.majority_undersampling:
            train_sampler = BalancedSampler(train_ds, labels=[item["label"] for item in datalist_train],
                                            sampling_strategy='under', distributed=True)
        else:
            train_sampler = Sampler(train_ds)
    else:
        if args.majority_undersampling:
            train_sampler = BalancedSampler(train_ds, labels=[item["label"] for item in datalist_train],
                                            sampling_strategy='under', distributed=False)
        else:
            train_sampler = None

    train_loader = data.ThreadDataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler
        )

    # Validation Dataloader
    val_ds = data.Dataset(data=datalist_val, transform=val_transform)
    
    if args.distributed:
        if args.majority_undersampling:
            val_sampler = BalancedSampler(val_ds, labels=[item["label"] for item in datalist_val],
                                            sampling_strategy='under', distributed=True)
        else:
            val_sampler = Sampler(val_ds)
    else:
        if args.majority_undersampling:
            val_sampler = BalancedSampler(val_ds, labels=[item["label"] for item in datalist_val],
                                            sampling_strategy='under', distributed=False)
        else:
            val_sampler = None
            
        
    val_loader = data.ThreadDataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=val_sampler)
    loader = [train_loader, val_loader]
    
    return loader