import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

img_path = 'cat.jpg'   
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0


def visualize_feature_maps(base_model, layer_name, model_name, save_name):

    print(f"\nLoading {model_name} ...")

    model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(layer_name).output
    )

    feature_maps = model.predict(img_array)

    print(f"{model_name} Feature map shape:", feature_maps.shape)

    num_channels = feature_maps.shape[-1]

 
    square = int(np.sqrt(num_channels))
    square = min(square, 8)   

    plt.figure(figsize=(8, 8))

    for i in range(square * square):
        plt.subplot(square, square, i + 1)
        plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
        plt.axis('off')

    plt.suptitle(f"{model_name} - {layer_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()

    print(f"{save_name} saved successfully.")



#  VGG16

vgg_model = VGG16(weights='imagenet', include_top=False)
visualize_feature_maps(
    base_model=vgg_model,
    layer_name='block1_conv1',
    model_name='VGG16',
    save_name='vgg16_featuremap.png'
)



#  ResNet50

resnet_model = ResNet50(weights='imagenet', include_top=False)
visualize_feature_maps(
    base_model=resnet_model,
    layer_name='conv1_relu',
    model_name='ResNet50',
    save_name='resnet50_featuremap.png'
)



# 3 InceptionV3

inception_model = InceptionV3(weights='imagenet', include_top=False)
visualize_feature_maps(
    base_model=inception_model,
    layer_name='mixed0',
    model_name='InceptionV3',
    save_name='inceptionv3_featuremap.png'
)

print("\nAll feature maps generated successfully!")
