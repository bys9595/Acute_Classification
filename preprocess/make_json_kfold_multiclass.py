import os
import json
import math
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import numpy as np
import natsort
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold # library name: iterative-stratification


# 데이터 폴더 경로 설정
years = ['2014', '2015', '2016', '2018', '2019', '2020_1', '2020_3', '2021_1', '2021_2', '2021_3', '2021_4', '2021_6', '2022_2023']
# target_classes = ["Acute appendicitis", "Acute cholecystitis", "Biliary stone", "Abdominal aortic aneurysm", 
#                   "Active bleeding", "Acute diverticulitis", "Acute pancreatitis", "Acute pyelonephritis", 
#                   "Ureter stone", "Mass (suspicious malignancy)", 
#                   "Aortic dissection", "Hemoperitoneum", "Abscess", "Bowel obstruction", "Pneumoperitoneum", "Epiploic appendagitis", "SMA lesion (thrombosis, dissection)", "Celiac lesion", "Hematoma", "Peptic ulcer disease", "Adrenal lesion (adenoma, hyperplasia)", "Normal"]
target_classes = ["Acute pyelonephritis", "Ureter stone", "Mass (suspicious malignancy)", "Bowel obstruction"]
task_name = '4Class+Normal'

# -----------------------------------------------------------------------------------------------
for year in years:
    print(f'Processing {year} ---------------------------------------------------------------------------------')
    data_folder = '/mai_nas/Private_Dataset/GangNam_SEV/' + year
    label_dir = '/mai_nas/Private_Dataset/GangNam_SEV/labels/' + year + '.xlsx'
    out_path = '/mai_nas/Private_Dataset/GangNam_SEV/jsons_multiclass/' + year + '/' + task_name + '/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    train_nii_files=[]
    val_nii_files=[]
    test_nii_files=[]

    total_nii_files = [f for f in os.listdir(data_folder) if f.endswith('.nii.gz')]
    total_nii_files = natsort.natsorted(total_nii_files)

    excel_file = pd.read_excel(label_dir)

    sampled_files = []
    sampled_labels = []

    for patient in total_nii_files:
        patient_id = patient.split('_')[0]
        date = patient.split('_')[1]
        
        # excel_file 에서 patient_id 와 date 가 일치하는 행을 찾는다.
        patient_info = excel_file[(excel_file['patient_id'] == int(patient_id)) & (excel_file['date'] == int(date))]
        
        target_class_labels = []
        for target_class in target_classes:
            target_class_label = patient_info[target_class].tolist()
            target_class_labels.append(target_class_label[0])
        
        # 뒤에다가 normal label 추가
        if sum(target_class_labels) == 0:
            target_class_labels.append(1)
        else:
            target_class_labels.append(0)
        
        sampled_files.append(patient)
        sampled_labels.append(target_class_labels)


    # 데이터 개수 확인
    print(f"Total Case: {len(total_nii_files)}")
    sampled_labels_numpy = np.array(sampled_labels)
    num_of_classes = sampled_labels_numpy.sum(0).tolist()

    # 각 class 별 데이터 개수 확인
    for i, target_class in enumerate(target_classes + ['Normal']):
        print(f"  - {target_class}: {num_of_classes[i]}")
    
    # numpy array로 변환
    sampled_labels = np.array(sampled_labels)
    
    # y.sum(1) 에서 1이 아닌 것들의 index 찾기
    multi_disease_indices = np.where(sampled_labels.sum(1) != 1)[0]
    
    # sampled files와 sampled labels 에서 multi_disease_indices 에 해당하는 것들을 제거
    sampled_files = [sampled_files[i] for i in range(len(sampled_files)) if i not in multi_disease_indices]
    sampled_labels = np.stack([sampled_labels[i] for i in range(len(sampled_labels)) if i not in multi_disease_indices], axis=0)
    sampled_labels = np.argmax(sampled_labels, axis=1).tolist()
    
    
    # msss = MultilabelStratifiedShuffleSplit(n_splits=5, test_size=1/6, random_state=42)
    # kf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    train_files, test_files, train_labels, test_labels = train_test_split(
        sampled_files, sampled_labels, test_size=1/6, random_state=42, stratify=sampled_labels if sum(sampled_labels) > 6 else None)

    # K-Fold 설정
    if sum(sampled_labels) > 5:
        # Positive label이 5개 이상인 경우 StratifiedKFold 설정
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        # Positive label이 5개 이하인 경우 KFold 설정
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

    
    # K-Fold Cross Validation
    folds = []

    for train_index, val_index in kf.split(train_files, np.array(train_labels)):
        train_data = np.array(train_files)[train_index].tolist()
        val_data = np.array(train_files)[val_index].tolist()
        train_labels_fold = np.array(train_labels)[train_index].tolist()
        val_labels_fold = np.array(train_labels)[val_index].tolist()
        
        folds.append({
            'training': train_data,
            'validation': val_data,
            'training_labels': train_labels_fold,
            'val_labels': val_labels_fold
        })

    for i, fold in enumerate(folds):
        # training 데이터 생성
        training_data = [{"image": f, 'label': l} for f, l in zip(fold['training'], fold['training_labels'])]
        validation_data = [{"image": f, 'label': l} for f, l in zip(fold['validation'], fold['val_labels'])]
        test_data = [{"image": f, 'label': l} for f, l in zip(test_files, test_labels)]
        
        # training_data의 label 은 0,1,2,3,4 로 구성되어있고, 0의 개수, 1의 개수, ... 4의 개수를 계산
        label_count_dict = {}
        print_class_names = target_classes + ['Normal']
        for label in fold['training_labels']:
            label_count_dict[print_class_names[label]] = label_count_dict.get(print_class_names[label], 0) + 1
        
        # label_count_dict 를 print_class_names 순서대로 정렬
        sorted_label_count_dict = {print_class_names[i]: label_count_dict[print_class_names[i]] for i in range(len(print_class_names))}
        print(f'Fold {i} : {sorted_label_count_dict}')
        
        # JSON 데이터 구조 생성
        data = {
            "description": "GangNam Severance Data",
            "year": year,
            "modality": {
                "0": "CT"
            },
            "numTraining": len(training_data),
            "numValidation": len(validation_data),
            "numTest": len(test_data),
            # sorted_label_count_dict 를 순서대로 0번, 1번을 붙이기, key 값도 보여주기
            "target_classes_and_count_of_training_data": {f"{i}" + " (" + key + ")": count for i, (key, count) in enumerate(sorted_label_count_dict.items())},
            "tensorImageSize": "3D",
            "training": training_data,
            "validation": validation_data,
            "test": test_data,
            "label_count": label_count_dict
        }

        # JSON 파일 생성
        out_name = os.path.join(out_path, 'dataset_GNSEV_' + year + '_fold'+ str(i) +'.json')
        with open(out_name, 'w') as f:
            json.dump(data, f, indent=4)

        print("JSON file created successfully!")