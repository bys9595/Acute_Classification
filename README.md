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


## Run
### Binary Classification
```
# $1 클래스 이름, $2 모델 이름
bash train_binary_multiGPU.sh acute_appendicitis yunsu
```

### Multi-Class Classification
```
bash train_multiclass_multiGPU.sh 4Class+Normal yunsu
```


### 한방에 돌리기
```
# $1 task 이름, $2 모델 이름
python multirun_binary.py
```
GPU number는 각 shell 파일에서 지정해주기!~
