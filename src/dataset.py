"""dataset.py
Utilities to load image dataset with tf.data and basic augmentation.
"""
import os
import tensorflow as tf




def get_datasets(data_dir, img_size=224, batch_size=32, val_split=0.2, seed=123):
"""Return (train_ds, val_ds, class_names)
Expects directory with subfolders per class (tf.keras.preprocessing style).
"""
data_dir = os.path.abspath(data_dir)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
data_dir,
validation_split=val_split,
subset="training",
seed=seed,
image_size=(img_size, img_size),
batch_size=batch_size
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
data_dir,
validation_split=val_split,
subset="validation",
seed=seed,
image_size=(img_size, img_size),
batch_size=batch_size
)


class_names = train_ds.class_names


# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


return train_ds, val_ds, class_names




def augmentation_layer(img_size=224):
return tf.keras.Sequential([
tf.keras.layers.RandomFlip("horizontal"),
tf.keras.layers.RandomRotation(0.15),
tf.keras.layers.RandomZoom(0.1),
tf.keras.layers.RandomContrast(0.1),
], name='augmentation')
