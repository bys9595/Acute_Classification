import os
import json
import math
from sklearn.model_selection import KFold
import numpy as np
import natsort
import pandas as pd

# 데이터 폴더 경로 설정
year = '2016'
test_size = 500

data_folder = '/mai_nas/Private_Dataset/GangNam_SEV/' + year
label_dir = '/mai_nas/Private_Dataset/GangNam_SEV/GNSEV_label_total.xlsx'
out_path = '/mai_nas/Private_Dataset/GangNam_SEV/'

train_nii_files=[]
val_nii_files=[]
test_nii_files=[]

total_nii_files = [f for f in os.listdir(data_folder) if f.endswith('.nii.gz')]
total_nii_files = natsort.natsorted(total_nii_files)


excel_file = pd.read_excel(label_dir)

ID = excel_file['ID'].tolist()
Normal = excel_file['Normal'].tolist()
Abnormal = [1- i for i in Normal]

# 전체 데이터 길이
total_length = len(total_nii_files)

# Train, Validation, Test 개수 설정
remaining_size = total_length - test_size


# 리스트를 섞어줌
np.random.shuffle(total_nii_files)

# 테스트 데이터 분리
testing_data = total_nii_files[:test_size]
remaining_data = total_nii_files[test_size:]


test_labels = []
for name in testing_data:
    pat_id = int(name.split('_')[0])
    idx = ID.index(pat_id)
    test_labels.append(Abnormal[idx])
    

# K-Fold 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# K-Fold Cross Validation
folds = []

for train_index, val_index in kf.split(remaining_data):
    train_labels = []
    val_labels = []
    train_data = np.array(remaining_data)[train_index].tolist()
    val_data = np.array(remaining_data)[val_index].tolist()
    
    for name in train_data:
        pat_id = int(name.split('_')[0])
        idx = ID.index(pat_id)
        train_labels.append(Abnormal[idx])
        
    for name in val_data:
        pat_id = int(name.split('_')[0])
        idx = ID.index(pat_id)
        val_labels.append(Abnormal[idx])
    
    folds.append({
        'training': train_data,
        'validation': val_data,
        'training_labels' : train_labels,
        'val_labels' : val_labels
    })
    
for i, fold in enumerate(folds):
    # training 데이터 생성
    training_data = [{"image": f, 'label': int(l)} for f, l in zip(fold['training'], fold['training_labels'])]
    validation_data = [{"image": f, 'label': int(l)} for f, l in zip(fold['validation'], fold['val_labels'])]
    test_data = [{"image": f, 'label': int(l)} for f, l in zip(testing_data, test_labels)]

    # TCIA Restricted
    # CC BY 3.0
    # JSON 데이터 구조 생성
    data = {
        "description": "GangNam Severance Data",
        "year" : year,
        # "licence": "CC BY-NC-ND 3.0",
        # "licence": "TCIA Restricted",
        "modality": {
            "0": "CT"
        },
        "numTraining": len(training_data),
        "numValidation": len(validation_data),
        "numTest": len(test_data),
        "tensorImageSize": "3D",
        "training": training_data,
        "validation": validation_data,
        "test": test_data,
    }

    # JSON 파일 생성
    out_name = os.path.join(out_path, 'dataset_GNSEV2016_fold'+ str(i) +'.json')
    with open(out_name, 'w') as f:
        json.dump(data, f, indent=4)

    print("JSON file created successfully!")
