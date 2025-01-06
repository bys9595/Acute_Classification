# Acute Classification


## Updates

***06/01/2025***
github upload

## Installation
1. torch install
```
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

2. requirements install
```
pip install -r requirements.txt
```


## 파일들 만들게된 흐름
- 처음에 `main.py`, `main_multiclass.py`, `eval.py`, `eval_multiclass.py` 를 만들었는데 threshold 설정하는것도 귀찮고 evaluation을 따로 하는게 귀찮았음
- 한방에 evaluation 될 수 있도록 `train_binary.sh`, `train_multiclass.sh` 을 만들었음
- multi gpu로 돌리면 빠르니까 `train_binary_multiGPU.sh`, `train_multiclass_multiGPU.sh` 도 만들었음
- 한 방에 여러 class 돌리고 싶어서 `multirun_binary.py`, `multirun_multiclass.py` 도 만들었음

## Run
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

 
