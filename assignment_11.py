import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ==========================================
# 1️⃣ Load MNIST (small subset for speed)
# ==========================================
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

# Take only 1000 samples for ultra fast run
x_train = x_train[:1000]
y_train = y_train[:1000]

# Convert to RGB and resize to 224x224
x_train = np.stack([x_train]*3, axis=-1)
x_train = tf.image.resize(x_train, (224,224))
x_train = x_train / 255.0

# ==========================================
# 2️⃣ Load Pretrained VGG16 (ImageNet)
# ==========================================
base_model = VGG16(weights='imagenet',
                   include_top=False,
                   input_shape=(224,224,3))

# Extract features BEFORE transfer learning
feature_extractor = models.Model(inputs=base_model.input,
                                 outputs=base_model.layers[-1].output)

features_before = feature_extractor.predict(x_train, verbose=0)
features_before = features_before.reshape(1000, -1)

print("Feature shape before:", features_before.shape)

# ==========================================
# 3️⃣ Dimensionality Reduction BEFORE
# ==========================================
pca = PCA(n_components=2)
pca_before = pca.fit_transform(features_before)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_before = tsne.fit_transform(features_before)

# Plot PCA BEFORE
plt.figure()
plt.scatter(pca_before[:,0], pca_before[:,1], c=y_train, cmap='tab10')
plt.title("PCA Before Transfer Learning")
plt.colorbar()
plt.savefig("pca_before.png")
plt.close()

# Plot t-SNE BEFORE
plt.figure()
plt.scatter(tsne_before[:,0], tsne_before[:,1], c=y_train, cmap='tab10')
plt.title("t-SNE Before Transfer Learning")
plt.colorbar()
plt.savefig("tsne_before.png")
plt.close()

# ==========================================
# 4️⃣ Transfer Learning (Ultra Light)
# Freeze base model
# ==========================================
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train only 2 epochs (fast)
model.fit(x_train, y_train,
          epochs=2,
          batch_size=32,
          verbose=1)

# ==========================================
# 5️⃣ Extract features AFTER transfer learning
# ==========================================
feature_extractor_after = models.Model(inputs=model.input,
                                       outputs=model.layers[-3].output)

features_after = feature_extractor_after.predict(x_train, verbose=0)

print("Feature shape after:", features_after.shape)

# ==========================================
# 6️⃣ Dimensionality Reduction AFTER
# ==========================================
pca_after = PCA(n_components=2).fit_transform(features_after)
tsne_after = TSNE(n_components=2, random_state=42, perplexity=30)\
                .fit_transform(features_after)

# Plot PCA AFTER
plt.figure()
plt.scatter(pca_after[:,0], pca_after[:,1], c=y_train, cmap='tab10')
plt.title("PCA After Transfer Learning")
plt.colorbar()
plt.savefig("pca_after.png")
plt.close()

# Plot t-SNE AFTER
plt.figure()
plt.scatter(tsne_after[:,0], tsne_after[:,1], c=y_train, cmap='tab10')
plt.title("t-SNE After Transfer Learning")
plt.colorbar()
plt.savefig("tsne_after.png")
plt.close()

print("\nAll 4 images saved successfully:")
print(" - pca_before.png")
print(" - tsne_before.png")
print(" - pca_after.png")
print(" - tsne_after.png")
