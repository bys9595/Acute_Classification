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
import torch
from utils.data_test import get_loader
from models.create_model import create_model
from torch.cuda.amp import autocast
import monai
from utils.metrics import BinaryEvalMetrics
import logging
from tqdm import tqdm
from timm.utils import setup_default_logging

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '28890'

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")

parser.add_argument("--logdir", default="/mai_nas/BYS/SSL/Finetune/MosMedData/runs/swint_swinmm_100k/", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--pretrained_dir", default="/mai_nas/BYS/SSL/Finetune/MosMedData/runs/swint_swinmm_100k/", type=str, help="pretrained checkpoint directory")
parser.add_argument("--pretrained_model_name", default="model_bestloss.pt", type=str, help="pretrained model name")

# parser.add_argument("--data_dir", default="/mai_nas/Private_Dataset/GangNam_SEV/2016/", type=str, help="dataset directory")
# parser.add_argument("--json_dir", default="/mai_nas/Private_Dataset/GangNam_SEV/jsons/2016/binary/", type=str, help="dataset directory")
parser.add_argument("--task_name", default="abdominal_aortic_aneurysm", type=str, help="task name")
parser.add_argument("--years", default=['2014', '2015', '2016', '2018', '2019', '2020_1', '2020_3',
                                        '2021_1', '2021_2', '2021_3', '2021_4', '2021_6', '2022_2023'], type=list, help="years")

parser.add_argument("--model", default="swint", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--batch_size", default=4, type=int, help="number of batch size")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=1, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=275, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--roi_x", default=192, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=192, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=64, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", default=False, help="use gradient checkpointing to save memory")

# additional custom
parser.add_argument("--test_data_key", default="validation", help="use squared Dice")
parser.add_argument("--best_thres", default=None, type=float, help="use squared Dice")
parser.add_argument('--fold_num', default='0', type=str, choices=['0', '1', '2', '3', '4'])


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    
    os.makedirs(args.logdir, exist_ok=True)
    setup_default_logging(log_path=os.path.join(args.logdir, args.test_data_key + '.txt'))
    
    logger = init_log('global', logging.INFO)
    logger.propagate = False

    loader = get_loader(args)

    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_pth = os.path.join(pretrained_dir, model_name)
    model = create_model(args)

    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict, strict=True)
    model.eval()
    model.to(device)
    
    test_metrics = BinaryEvalMetrics()
    
    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(loader)):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]

            target = target.float()

            data, target = data.cuda(), target.cuda()

            with autocast(enabled=args.amp):
                logits = model(data)
                if isinstance(logits, monai.data.meta_tensor.MetaTensor):
                    logits = logits.as_tensor()
            
            test_metrics.update(logits, target)
    
    print('Compute evaluation metrics')
    metrics_dict = test_metrics.on_epoch_end_compute(args.best_thres)
    
    sorteddict = list(metrics_dict.keys())
    sorteddict.sort()
    metrics_dict = {i: metrics_dict[i] for i in sorteddict}
    
    logging.info('From ' + args.pretrained_model_name)
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            if key in ['Accuracy', 'Sensitivity', 'Specificity']:
                logging.info(str(key)+ "  :  " + str(round(value*100, 2)))
            elif key in ['Best_thres']:
                logging.info(str(key)+ "  :  {}".format(str(value)))
            else:
                logging.info(str(key)+ "  :  " + str(round(value, 3)))
        else:
            logging.info(str(key)+ "  :  {}".format(value))
            
    
    print("RETURN: " + str(metrics_dict['Best_thres'])) # Do not remove this print line, this is for capturing the path of model


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
