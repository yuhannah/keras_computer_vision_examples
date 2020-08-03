import tensorflow as tf
import matplotlib.pyplot as plt
import os

print("Tensorflow Version: ",tf.__version__)
print("Tensorflow.keras Version: ",tf.keras.__version__)



# 过滤掉头部没有字符串“JFIF”的编码错误的图像
num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join("PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)

# 生成数据集
image_size = (180, 180)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)


# 显示数据
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

# 数据增强：翻转、旋转
data_augmentation = tf.keras.Sequential(
    [
        tf.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

# 显示翻转的图片
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

# 归一化图像数据
augmented_train_ds = train_ds.map(
  lambda x, y: (data_augmentation(x, training=True), y))

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

# 构建模型
def make_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = tf.layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = tf.layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = tf.layers.BatchNormalization()(x)
    x = tf.layers.Activation("relu")(x)

    x = tf.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.layers.BatchNormalization()(x)
    x = tf.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = tf.layers.Activation("relu")(x)
        x = tf.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = tf.layers.BatchNormalization()(x)

        x = tf.layers.Activation("relu")(x)
        x = tf.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = tf.layers.BatchNormalization()(x)

        x = tf.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = tf.layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = tf.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = tf.layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = tf.layers.BatchNormalization()(x)
    x = tf.layers.Activation("relu")(x)

    x = tf.layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = tf.layers.Dropout(0.5)(x)
    outputs = tf.layers.Dense(units, activation=activation)(x)
    return tf.keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
tf.keras.utils.plot_model(model, show_shapes=True)

# 训练模型
epochs = 50
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)

