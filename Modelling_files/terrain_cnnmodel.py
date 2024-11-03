import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths
data_dir = 'C:/Users/GOKUL RAJ/RAJ/TERRAIN ANALYSIS/terrain_dataset'  # Path to your dataset

# Set up image data generator
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
    validation_split=0.2  # Use 20% of data for validation
)

# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),  # Resize images to this size
    batch_size=32,
    class_mode='categorical',  # Use categorical labels for multi-class classification
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Use categorical labels for multi-class classification
    subset='validation',
    shuffle=False  # Important for evaluation
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Prevent overfitting
    Dense(len(train_generator.class_indices), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,  # Adjust the number of epochs as needed
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Save the model
model.save("terrain_classifier_cnnmodel.h5")

# Evaluation metrics
# 1. Generate predictions on the validation set
validation_generator.reset()  # Ensure no shuffling in predictions
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = validation_generator.classes

# 2. Classification report
class_labels = list(validation_generator.class_indices.keys())
report = classification_report(y_true, y_pred_classes, target_names=class_labels)
print("Classification Report:\n", report)

# 3. Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()

# Function to predict the terrain type
def predict_terrain(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image
    predictions = model.predict(img_array)
    predicted_class = train_generator.class_indices
    predicted_label = list(predicted_class.keys())[np.argmax(predictions)]
    print(f"The predicted terrain type is: {predicted_label}")

# Example usage
# Replace 'path_to_your_image' with the actual image path you want to predict
predict_terrain('C:/Users/GOKUL RAJ/RAJ/TERRAIN ANALYSIS/terrain_dataset/Grass/Grass_aug_4.jpg')
