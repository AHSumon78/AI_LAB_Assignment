import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# ======================================
# Load CIFAR-10 (Offline Dataset)
# ======================================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Make it Binary (Class 0 vs Class 1)
# 0 = airplane
# 1 = automobile
train_filter = np.where((y_train == 0) | (y_train == 1))[0]
test_filter = np.where((y_test == 0) | (y_test == 1))[0]

x_train, y_train = x_train[train_filter], y_train[train_filter]
x_test, y_test = x_test[test_filter], y_test[test_filter]

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Resize to 224x224 for VGG16
x_train = tf.image.resize(x_train, (224,224))
x_test = tf.image.resize(x_test, (224,224))

# ======================================
# Build Model Function
# ======================================
def build_model(mode):

    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(224,224,3))

    if mode == "feature_extraction":
        for layer in base_model.layers:
            layer.trainable = False

    elif mode == "full_finetune":
        for layer in base_model.layers:
            layer.trainable = True

    elif mode == "partial_finetune":
        for layer in base_model.layers[:15]:
            layer.trainable = False
        for layer in base_model.layers[15:]:
            layer.trainable = True

    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# ======================================
# Choose Mode
# ======================================
mode = "partial_finetune"
# Try:
# "feature_extraction"
# "full_finetune"

model = build_model(mode)

# ======================================
# Train
# ======================================
history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=32
)

# ======================================
# Save Accuracy Plot
# ======================================
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title(f'Accuracy ({mode})')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("accuracy_plot.png")
plt.close()

# ======================================
# Save Loss Plot
# ======================================
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'Loss ({mode})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")
plt.close()

print("\nTraining Complete (Offline Mode)")
print("Saved:")
print(" - accuracy_plot.png")
print(" - loss_plot.png")
