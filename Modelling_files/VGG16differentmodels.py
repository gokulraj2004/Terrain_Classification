import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to your dataset
dataset_path = 'C:/Users/GOKUL RAJ/RAJ/TERRAIN ANALYSIS/terrain_dataset'  # Replace with your actual path

# Check the directory structure and count images in each class
for class_name in os.listdir(dataset_path):
    class_folder = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_folder):
        image_count = len([f for f in os.listdir(class_folder) if f.endswith(('.jpg', '.png'))])
        print(f"{class_name}: {image_count} images")

# Set up the ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Use 20% of the data for validation
)

# Create training data generator
train_generator = train_datagen.flow_from_directory(
    directory=dataset_path,
    target_size=(150, 150),  # Resize images to this size
    batch_size=32,
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    subset='training'  # Set as training data
)

# Create validation data generator
validation_generator = train_datagen.flow_from_directory(
    directory=dataset_path,
    target_size=(150, 150),  # Resize images to this size
    batch_size=32,
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    subset='validation'  # Set as validation data
)

# Define your model (e.g., VGG16)
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers on top
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')  # Adjust the number of output classes as needed
])

# Compile the model with additional metrics
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10  # Adjust the number of epochs as needed
)

# Save the model (optional)
model.save('terrain_analysis_vgg16model.h5')

# Evaluate on the validation set
val_generator = validation_generator
val_generator.reset()
predictions = model.predict(val_generator, steps=val_generator.samples // val_generator.batch_size + 1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_generator.classes[:len(predicted_classes)]

# Classification report
class_labels = list(val_generator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print("Classification Report:\n", report)

# Confusion Matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
plt.show()

# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Load the image and resize it
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    img_array /= 255.0  # Scale pixel values
    return img_array

# Path to your test image
test_image_path = "C:/Users/GOKUL RAJ/RAJ/TERRAIN ANALYSIS/terrain_dataset/Grass/Grass_aug_4.jpg"

# Load and preprocess the test image
test_image = load_and_preprocess_image(test_image_path)

# Make predictions
predictions = model.predict(test_image)

# Get the predicted class index
predicted_class_index = np.argmax(predictions, axis=1)[0]

# Get class labels from the training generator
class_labels = train_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}  # Invert the dictionary to map indices to class names

# Print the predicted class
predicted_class = class_labels[predicted_class_index]
print(f"The predicted terrain type is: {predicted_class}")
