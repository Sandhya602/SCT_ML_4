# SCT_ML_4
# Hand Gesture Recognition


This repository trains a hand gesture recognition model that classifies images or live webcam frames into gesture labels. It uses transfer learning with MobileNetV2 and a small classification head. The repo also includes a webcam demo for live inference.


## Dataset


Place your dataset under `data/gestures/` with one folder per gesture label, e.g.:
data/gestures/ ├── thumbs_up/ │ ├── img001.jpg │ └── ... ├── palm/ ├── fist/ └── peace/
Images should ideally be RGB, variety of backgrounds, lighting, and hand orientations for robustness.


If you don't have a dataset, you can record a short video and extract frames (not included here).


## Installation


```bash
python -m venv venv
source venv/bin/activate # macOS / Linux
venv\Scripts\activate # Windows
pip install -r requirements.txt
