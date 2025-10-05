"""model.py
Builds a transfer-learning model using MobileNetV2.
"""
import tensorflow as tf




def build_model(num_classes, img_size=224, base_trainable=False):
base_model = tf.keras.applications.MobileNetV2(
input_shape=(img_size, img_size, 3),
include_top=False,
weights='imagenet'
)
base_model.trainable = base_trainable


inputs = tf.keras.Input(shape=(img_size, img_size, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)


model = tf.keras.Model(inputs, outputs)
return model
