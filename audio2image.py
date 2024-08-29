import librosa
import librosa.display
import struct
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import json

def readMarkers(filePath):
    _, audioType = filePath.split('.')

    if audioType == 'wav':
        cue_lst = readWavMarkers(filePath)
    elif audioType == 'mp3':
        cue_lst = readMP3Markers(filePath)
    else:
        raise Exception("音频格式不支持！")

    return cue_lst

def readWavMarkers(filePath):
    if hasattr(filePath, 'read'):
        fid = filePath
    else:
        fid = open(filePath, 'rb')
    fsize = _read_riff_chunk(fid)
    cue = []
    while fid.tell() < fsize:
        chunk_id = fid.read(4)
        if chunk_id == b'cue ':
            size, numcue = struct.unpack('<ii', fid.read(8))
            for c in range(numcue):
                id, position, datachunkid, chunkstart, blockstart, sampleoffset = struct.unpack('<iiiiii', fid.read(24))
                cue.append(position)
        else:
            _skip_unknown_chunk(fid)
    fid.close()
    return cue

def readMP3Markers(filePath):
    cue = []
    metadata = audio_metadata.load(filePath)
    try:
        num_chunk = len(str(metadata.tags['private'][0]).split('xmpDM:startTime="'))
    except:
        return []
    for i in range(num_chunk):
        if i == 0:
            continue
        else:
            marker = str(metadata.tags['private'][0]).split('xmpDM:startTime="')[i]
            marker = int(marker.split('"')[0])
            cue.append(marker)
    return cue

def plotMarkers(filePath):
    sig, _ = librosa.load(filePath, sr=None)
    cue_lst = readWavMarkers(filePath)
    plt.plot(sig)
    for x in cue_lst:
        plt.axvline(x, c='r')
    plt.show()

def _read_riff_chunk(fid):
    str1 = fid.read(4)
    if str1 != b'RIFF':
        raise ValueError("Not a WAV file.")
    fsize = struct.unpack('<I', fid.read(4))[0] + 8
    str2 = fid.read(4)
    if str2 != b'WAVE':
        raise ValueError("Not a WAV file.")
    return fsize

def _skip_unknown_chunk(fid):
    data = fid.read(4)
    size = struct.unpack('<i', data)[0]
    if bool(size & 1):
        size += 1
    fid.seek(size, 1)

def audio2image(audio_sig, savePath, sr, filename):
    S = librosa.feature.melspectrogram(y=audio_sig, sr=sr, n_mels=128, n_fft=2048, hop_length=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    image_height, image_width = S_dB.shape
    print(S_dB.shape)
    factor = 11
    dpi = 55 / factor
    plt.figure(figsize=(image_width / dpi, image_height / dpi))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.savefig(f"{savePath}/{filename}.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    return image_width * factor, image_height * factor

def generate_melspectrogram_and_annotation(filePath, savePath, annotations, image_id, annotation_id, target_sr=16000):
    _, ori_sr = librosa.load(filePath, sr=None)
    wav_signal, _ = librosa.load(filePath, sr=target_sr)
    filename = os.path.basename(filePath)[:-4]
    try:
        image_width, image_height = audio2image(wav_signal, savePath, target_sr, filename)
    except ValueError as e:
        if str(e) == 'Image size of pixels is too large.':
            print(f"Error processing file {filePath}: {e}")
            return annotations, annotation_id

    duration = len(wav_signal) / target_sr
    true_start_end_points = readMarkers(filePath)
    true_start_end_points = [int(point / ori_sr * target_sr) for point in true_start_end_points]

    if len(true_start_end_points) % 2 != 0:
        print(f"Error: {filePath} has an odd number of segmentation points: {true_start_end_points}")
        return annotations, annotation_id
    
    for i in range(0, len(true_start_end_points), 2):
        start_time = true_start_end_points[i] / target_sr
        end_time = true_start_end_points[i + 1] / target_sr
        bbox = [start_time / duration * image_width, 0, (end_time - start_time) / duration * image_width, image_height]

        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "iscrowd": 0,
            "area": bbox[2] * bbox[3],
            "bbox": bbox,
            "segmentation": [],
        }
        annotations["annotations"].append(annotation)
        annotation_id += 1

    return annotations, annotation_id

if __name__ == '__main__':
    data_root = "/home/jovyan/work/MusicYOLO/data4retrain/train/subdir32"
    melspec_save_path = "/home/jovyan/work/MusicYOLO/mmmel/train32"
    os.makedirs(melspec_save_path, exist_ok=True)
    target_sr = 16000
    allWavFiles = glob.glob(f"{data_root}/*.wav")
    num_wav_files = len(allWavFiles)
    print(f"Number of wav files: {num_wav_files}")
    coco_annotations = {
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "note",
                "supercategory": "Cancer"
            }
        ]
    }
    image_id = 1
    annotation_id = 1
    for wav_filepath in allWavFiles:
        print(f'processing file {wav_filepath}...')
        try:
            coco_annotations, annotation_id = generate_melspectrogram_and_annotation(wav_filepath, melspec_save_path, coco_annotations, image_id, annotation_id, target_sr=target_sr)
        except Exception as e:
            print(f"Error processing file {wav_filepath}: {e}")
        image_id += 1

    annotations_save_path = "/home/jovyan/work/MusicYOLO/mmmel/32train_annotations.json"
    with open(annotations_save_path, 'w') as f:
        json.dump(coco_annotations, f, indent=4)