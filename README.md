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
bash train_binary_multiGPU.sh
```

### Multi-Class Classification
```
bash train_multiclass_multiGPU.sh
```

GPU number는 각 shell 파일에서 지정해주기!~
