import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import numpy as np


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)


def build_cnn(activation='relu', loss_fn='categorical_crossentropy'):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation=activation, input_shape=(32,32,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation=activation))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (3,3), activation=activation))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=activation))
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model

activations = ['relu', 'sigmoid', 'tanh']
loss_fn = 'categorical_crossentropy'

for act in activations:
    print(f"\nTraining with activation: {act}")
    model = build_cnn(activation=act, loss_fn=loss_fn)
    history = model.fit(x_train, y_train_cat,
                        validation_data=(x_test, y_test_cat),
                        epochs=5,
                        batch_size=64,
                        verbose=1)
    
    # Save accuracy plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"Accuracy - Activation: {act}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"accuracy_{act}.png")
    plt.close()


print("\nTraining with loss function: MSE")
model = build_cnn(activation='relu', loss_fn='mean_squared_error')
history = model.fit(x_train, y_train_cat,
                    validation_data=(x_test, y_test_cat),
                    epochs=5,
                    batch_size=64,
                    verbose=1)

# Save loss plot
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss - Loss Function: MSE")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_mse.png")
plt.close()

print("\nAll plots saved:")
print(" - accuracy_relu.png")
print(" - accuracy_sigmoid.png")
print(" - accuracy_tanh.png")
print(" - loss_mse.png")
