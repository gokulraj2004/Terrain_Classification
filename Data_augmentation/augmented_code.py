import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np

# Directory paths
input_folder = "C:/Users/GOKUL RAJ/RAJ/Github codes/Terrain_Classification/Textures_used"  # Folder containing original images
output_folder = "C:/Users/GOKUL RAJ/RAJ/Github codes/Terrain_Classification/new_terrain_dataset"  # Folder to save augmented images

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize ImageDataGenerator with augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Set to generate 25 augmented images per original image
augmented_images_per_label = 25

# Loop through each label folder
for label in ['sand', 'soil', 'asphalt', 'concrete', 'grass']:
    label_folder = os.path.join(input_folder, label)
    output_label_folder = os.path.join(output_folder, label)
    os.makedirs(output_label_folder, exist_ok=True)

    # Loop through each image in the label folder
    for image_file in os.listdir(label_folder):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):  # Ensure it's an image
            image_path = os.path.join(label_folder, image_file)
            img = load_img(image_path)  # Load image
            x = img_to_array(img)  # Convert to numpy array
            x = np.expand_dims(x, axis=0)  # Reshape for the generator

            # Create 25 augmented images per original image
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=output_label_folder,
                                      save_prefix=f'{label}_aug', save_format='jpg'):
                i += 1
                if i >= augmented_images_per_label:
                    break  # Stop after generating 25 images per original

print("Data augmentation completed with 25 images per original image.")
