"""evaluate.py
Evaluate a trained model on the validation set and print confusion matrix.
"""
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from dataset import get_datasets




def main():
p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--data_dir', required=True)
p.add_argument('--img_size', type=int, default=224)
args = p.parse_args()


model = tf.keras.models.load_model(args.model)
train_ds, val_ds, class_names = get_datasets(args.data_dir, img_size=args.img_size)


# gather true and pred
y_true = []
y_pred = []
for x_batch, y_batch in val_ds:
preds = model.predict(x_batch)
preds_labels = np.argmax(preds, axis=1)
y_true.extend(y_batch.numpy().tolist())
y_pred.extend(preds_labels.tolist())


print(classification_report(y_true, y_pred, target_names=class_names))
cm = confusion_matrix(y_true, y_pred)
print('Confusion matrix:\n', cm)




if __name__ == '__main__':
main()
