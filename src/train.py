"""train.py
def parse_args():
p = argparse.ArgumentParser()
p.add_argument('--data_dir', required=True, help='path to gestures dataset directory')
p.add_argument('--epochs', type=int, default=20)
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--img_size', type=int, default=224)
p.add_argument('--output', default='checkpoints/best_model.h5')
p.add_argument('--learning_rate', type=float, default=1e-4)
p.add_argument('--fine_tune', action='store_true', help='unfreeze base model after initial training')
return p.parse_args()




def main():
args = parse_args()
os.makedirs(os.path.dirname(args.output), exist_ok=True)


train_ds, val_ds, class_names = get_datasets(args.data_dir, img_size=args.img_size,
batch_size=args.batch_size)


num_classes = len(class_names)
model = build_model(num_classes, img_size=args.img_size, base_trainable=False)


# add augmentation on top of dataset pipeline or inside model
data_augmentation = augmentation_layer(args.img_size)


model.compile(
optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
loss='sparse_categorical_crossentropy',
metrics=['accuracy']
)


callbacks = [
tf.keras.callbacks.ModelCheckpoint(args.output, save_best_only=True, monitor='val_accuracy', mode='max'),
tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
]


# optionally map augmentation
train_ds_aug = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))


history = model.fit(
train_ds_aug,
validation_data=val_ds,
epochs=args.epochs,
callbacks=callbacks
)


# Optional fine-tuning
if args.fine_tune:
base_model = model.layers[2] # assuming ordering Input->preprocess->base
base_model.trainable = True
model.compile(
optimizer=tf.keras.optimizers.Adam(1e-5),
loss='sparse_categorical_crossentropy',
metrics=['accuracy']
)
model.fit(train_ds_aug, validation_data=val_ds, epochs=5)


print(f"Trained model saved to {args.output}")




if __name__ == '__main__':
main()
