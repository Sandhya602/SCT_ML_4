"""predict.py
Predict single image or all images in a folder using trained model.
"""
import argparse
import tensorflow as tf
import numpy as np
import cv2
import os




def load_and_preprocess(path, img_size):
img = cv2.imread(path)
if img is None:
raise ValueError(f"Cannot read image {path}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (img_size, img_size))
img = tf.keras.applications.mobilenet_v2.preprocess_input(img.astype('float32'))
return img




def predict_image(model, path, img_size, class_names):
img = load_and_preprocess(path, img_size)
preds = model.predict(np.expand_dims(img, axis=0))
idx = int(np.argmax(preds))
prob = float(np.max(preds))
return class_names[idx], prob




def main():
p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--image')
p.add_argument('--folder')
p.add_argument('--img_size', type=int, default=224)
args = p.parse_args()


model = tf.keras.models.load_model(args.model)
# infer class names from training data? user must supply same structure; we try to read from model if saved
# For simplicity, require that a file 'class_names.txt' exists next to model
class_names_path = os.path.splitext(args.model)[0] + '.class_names.txt'
if os.path.exists(class_names_path):
with open(class_names_path, 'r') as f:
class_names = [l.strip() for l in f.readlines()]
else:
# fallback: numeric names
class_names = [str(i) for i in range(model.output_shape[-1])]


if args.image:
label, prob = predict_image(model, args.image, args.img_si
