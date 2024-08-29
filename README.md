# CoughYOLO-Detection
Detection and identification of cough onset points based on [MusicYOLO](https://github.com/xk-wang/MusicYOLO/tree/main). We trained and evaluated the model on the highQ dataset and NewCough dataset.

## Clone the Repo
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
## Dataset Information
We trained the model using the highQ dataset and NewCough Dataset. These two datasets consist of approximately 10-second audio clips, each containing cough sounds collected from hospitals. The audio in the dataset is in .wav format. Sorry for we do not offer the dataset.
```
-highQ Dataset
--Train: 1136
--Valid: 200
--Test: 200 (same as validation)

-NewCough Dataset
--Train: 2341
--Valid: 486
--Test: 490 (different from validation)
```
## Data Preprocessing
Step 1. Process the WAV audio files in the dataset, convert them into Mel spectrograms, and generate the corresponding COCO format annotation files. 

Note: The original audio files are annotated with the start and end points of the cough sounds, so the annotation information for each audio file should be in pairs. However, some audio files have the issue where `true_start_end_points` are not in pairs. Our code has detected and removed these files.
`python3 audio2image.py`

Step 2. 

Step 3. Divide the dataset into training, testing, and validation sets.
Division rule: Ensure that audio from the same patient is not assigned to different groups.
`divide_dataset.ipynb`

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

