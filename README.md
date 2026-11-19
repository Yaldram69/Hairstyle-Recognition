# Hairstyle Recognition

YOLO instance segmentation for hairstyles. The trained model detects four classes (`curly`, `kinky`, `straight`, `wavy`) in images or from a webcam. The scripts keep only those four classes, discard other predictions, and draw the masks and boxes. The training notebook uses YOLO11m-seg on a Roboflow dataset.

## Features

- Instance segmentation of hairstyles with Ultralytics YOLO
- Output filtered to the four hairstyle classes
- Single-image inference with matplotlib display
- Webcam inference with a live window
- Batch display of several test images in a grid (notebook)

## Project structure

- `best (1).pt`, `last (1).pt`: trained checkpoints from training (best and last epoch). Both are about 45 MB.
- `CODE.PY.py`: single-image script. Defines `predict_hairstyles_simple(image_path, model_path, conf_threshold=0.5)` and shows the result with matplotlib.
- `web test.py`: webcam script. Defines `predict_hairstyles_webcam`. Press `q` to quit.
- `HairStyle__Recognition.ipynb`: inference notebook for several test images, shown in a 2x3 grid. Written for Google Colab (`/content/` paths).
- `testing_with_web_cam.ipynb`: notebook version of the webcam script.
- `Training_Code.ipynb`: downloads the dataset from Roboflow (workspace `hr-wh8o1`, project `hsrs-2`, version 1, YOLOv11 format), trains `yolo11m-seg.pt` for 30 epochs at image size 640, then reloads `runs/segment/train/weights/last.pt`.
- `Testing_images/`: sample images (curly, kinky, straight, wavy).

## Requirements

Imports used: `ultralytics`, `cv2`, `matplotlib`. Training also uses `roboflow`.

```
pip install ultralytics opencv-python matplotlib
pip install roboflow    # only for Training_Code.ipynb
```

## Installation

```
git clone https://github.com/Yaldram69/Hairstyle-Recognition.git
cd Hairstyle-Recognition
pip install ultralytics opencv-python matplotlib
```

## Usage

Scripts contain absolute paths from the author's machine. Edit the paths before running.

Single image, `CODE.PY.py`: set `image_path` and `model_path` in the call at the bottom of the file (or call the function from Python), for example:

```python
from CODE_PY import predict_hairstyles_simple  # or run the file directly after editing paths
predict_hairstyles_simple("Testing_images/curly.jpeg", model_path="best (1).pt", conf_threshold=0.5)
```

Input: a path to an image file. Output: a matplotlib window with the filtered annotations.

Webcam, `web test.py`: edit the hardcoded `YOLO(...)` path at the top of the file, then run:

```
python "web test.py"
```

Input: webcam index 0. Press `q` to quit.

## Notes

- The module-level `YOLO(...)` call in `web test.py` runs on import, so the model path there must be valid before the function is called.
- The default argument of `predict_hairstyles_simple` in `CODE.PY.py` is malformed (`r':\Users\...`) and must be replaced with a valid path.
- The class names expected by the scripts are `curly`, `kinky`, `straight` and `wavy`. If the checkpoint uses other names, no detections are kept.
- Accuracy is not evaluated by the code. The training notebook shows a confusion matrix and validation batch predictions, which are not included in this folder.
- The Roboflow dataset is not included. Training requires a Roboflow API key.
