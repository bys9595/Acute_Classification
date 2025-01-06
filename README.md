# Acute Classification


## Updates

***06/01/2025***
github upload

## Installation
1. conda env 만들기
```
conda create -n acute python=3.10 -y
```

2. torch install
```
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

3. requirements install
```
pip install -r requirements.txt
```


## 파일들 만들게된 흐름
- 처음에 `main.py`, `main_multiclass.py`, `eval.py`, `eval_multiclass.py` 를 만들었는데 threshold 설정하는것도 귀찮고 evaluation을 따로 하는게 귀찮았음
- 한방에 evaluation 될 수 있도록 `train_binary.sh`, `train_multiclass.sh` 을 만들었음
- multi gpu로 돌리면 빠르니까 `train_binary_multiGPU.sh`, `train_multiclass_multiGPU.sh` 도 만들었음
- 한 방에 여러 class 혹은 여러 task 가 queue 되어서 실행되게 만들고 싶어서 `multirun_binary.py`, `multirun_multiclass.py` 도 만들었음

## Run

bash 파일안에 있는 root_dir 경로 수정해주기!!!!!!!
### Binary Classification
```
# $1 class name $2 save directory name 
bash train_binary_multiGPU.sh acute_appendicitis yunsu
```

### Multi-Class Classification
```
# $1 task name $2 save directory name $3 num_classes
bash train_multiclass_multiGPU.sh 4Class+Normal yunsu 5
```


### 한방에 돌리기
```
python multirun_binary.py
python multirun_multiclass.py
```
GPU number는 각 shell 파일에서 지정해주기!~


## 새로운 Task 생성
multiclass 인 경우에만 해당됨.

1. `preprocess/make_json_kfold_multiclass.py` 파일에서 원하는 class 를 조합하고, task name 지정해서 json 파일 생성하기
2. `/mai_nas/Private_Dataset/GangNam_SEV/jsons_multiclass/` 폴더 안에 생성된 json 파일 확인하기
3. `train_multiclass_multiGPU.sh` 파일 실행할 때 지정한 task name을 argument로 넣어서 실행

모르겠으면 물어보슈

 