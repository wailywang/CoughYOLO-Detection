# CoughYOLO-Detection
Detection and identification of cough onset points based on [MusicYOLO](https://github.com/xk-wang/MusicYOLO/tree/main). We trained and evaluated the model on the highQ dataset and NewCough dataset.

## Clone the repo:
`git clone https://github.com/wailywang/CoughYOLO-Detection.git`

## Installation
Recommended pre-installed image: ubuntu20.04-cuda11.3.0-py37-torch1.11.0-tf1.15.5-1.6.1
```
cd CoughYOLO-Detection
python -m venv yolo
source yolo/bin/activate
pip install torch==1.11.0 torchvision==0.12.0
pip3 install -U pip && pip3 install -r requirements.txt
pip3 install -v -e .
cd ./apex
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" .
pip3 install cython
cd ..
cd ./cocoapi/PythonAPI
pip3 install -v .
```

## Mel Spectrogram Dataset Structure
```
datasets/
├── images/
│   ├── valid/
│   │   ├── mel1.png
│   │   ├── mel2.png
│   │   └── _annotations.coco.json
│   ├── train/
│   │   ├── mel1.png
│   │   ├── mel2.png
│   │   └── _annotations.coco.json
└── tests/
    ├── mel1.png
    ├── mel2.png
    └── _annotations.coco.json
```

