import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
import numpy as np

# ==========================================
# 1️⃣ Load CIFAR-10 Dataset
# ==========================================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# ==========================================
# 2️⃣ Data Augmentation Generator
# ==========================================
aug_gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# ==========================================
# 3️⃣ Function to Build CNN
# ==========================================
def build_cnn(use_dropout=False):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    if use_dropout:
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ==========================================
# 4️⃣ Train 4 Cases
# ==========================================
cases = [
    {"name":"case1_baseline", "dropout":False, "augment":False},
    {"name":"case2_dropout", "dropout":True, "augment":False},
    {"name":"case3_augment", "dropout":False, "augment":True},
    {"name":"case4_dropout_augment", "dropout":True, "augment":True}
]

for case in cases:
    print(f"\nTraining {case['name']}...")
    model = build_cnn(use_dropout=case['dropout'])
    
    if case['augment']:
        history = model.fit(
            aug_gen.flow(x_train, y_train, batch_size=64),
            validation_data=(x_test, y_test),
            epochs=5,
            verbose=1
        )
    else:
        history = model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            batch_size=64,
            epochs=5,
            verbose=1
        )
    
    # Save Accuracy Plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"Accuracy {case['name']}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{case['name']}_accuracy.png")
    plt.close()
    
    # Save Loss Plot
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"Loss {case['name']}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{case['name']}_loss.png")
    plt.close()
    
print("\nAll plots saved successfully!")
