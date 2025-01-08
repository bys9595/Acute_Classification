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
import itertools as it
from monai import data, transforms
from monai.data import *
from collections import Counter

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
    datalist_val = []
    for json_path, base_dir in zip(jsonlist, datadir_list):
        # Test Data List
        datalist_val_i = load_decathlon_datalist(json_path, False, args.test_data_key, base_dir=base_dir)
        for item in datalist_val_i:
            datalist_val.append({"image": item["image"], "label": item["label"]})

    print("Test Dataset: number of data: {}".format(len(datalist_val)))
    
    
    test_transform = transforms.Compose(
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
    
    test_ds = data.Dataset(data=datalist_val, transform=test_transform)    
    test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
    test_loader = data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=test_sampler, pin_memory=False)
    loader = test_loader
        
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
    datalist_val = []
    for json_path, base_dir in zip(jsonlist, datadir_list):
        # Test Data List
        datalist_val_i = load_decathlon_datalist(json_path, False, args.test_data_key, base_dir=base_dir)
        for item in datalist_val_i:
            datalist_val.append({"image": item["image"], "label": item["label"]})

    print("Test Dataset: number of data: {}".format(len(datalist_val)))
    label_counts = Counter(item["label"] for item in datalist_val)
    print("Class distribution: {}".format(label_counts))
    
    
    test_transform = transforms.Compose(
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
    
    test_ds = data.Dataset(data=datalist_val, transform=test_transform)    
    test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
    test_loader = data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=test_sampler, pin_memory=False)
    loader = test_loader
        
    return loader
