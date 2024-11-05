import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Terrain Classification",
    page_icon="🌍",
    layout="wide"
)

# Function to load models
@st.cache_resource
def load_models():
    cnn_model = tf.keras.models.load_model('C:/Users/GOKUL RAJ/RAJ/Github codes/Terrain_Classification/Saved_Models/terrain_classifier_cnnmodel.h5')
    vgg_model = tf.keras.models.load_model('C:/Users/GOKUL RAJ/RAJ/Github codes/Terrain_Classification/Saved_Models/terrain_analysis_vgg16model.h5')
    return cnn_model, vgg_model

# Function to preprocess image for CNN model
def preprocess_image_cnn(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Function to preprocess image for VGG16 model
def preprocess_image_vgg(image):
    img = image.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Function to get prediction
def get_prediction(model, preprocessed_image, class_labels):
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_index])
    predicted_class = class_labels[predicted_class_index]
    return predicted_class, confidence

# Function to get terrain emoji
def get_terrain_emoji(terrain_type):
    emoji_map = {
        'Asphalt': '🛣️',
        'Concrete': '🏗️',
        'Grass': '🌱',
        'Sand': '🏖️',
        'Soil': '🌰'
    }
    return emoji_map.get(terrain_type, '🌍')

def main():
    # Title and description
    st.title("🌍 Terrain Classification System")
    st.markdown("""
    This application analyzes surface images to identify different types of terrain:
    - **Asphalt** 🛣️ 
    - **Concrete** 🏗️ 
    - **Grass** 🌱 
    - **Sand** 🏖️ 
    - **Soil** 🌰 
    """)

    # Load models
    try:
        cnn_model, vgg_model = load_models()
        # Set up class labels for displaying predictions
        class_labels = {0: 'Asphalt', 1: 'Concrete', 2: 'Grass', 3: 'Sand', 4: 'Soil'}
        
        # File uploader
        uploaded_file = st.file_uploader("Upload a terrain image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption='Uploaded Terrain Image', use_column_width=True)
            
            # Make predictions
            with st.spinner('Analyzing terrain...'):
                # Preprocess image for both models
                preprocessed_image_cnn = preprocess_image_cnn(image)
                preprocessed_image_vgg = preprocess_image_vgg(image)
                
                # Get predictions from both models
                cnn_class, cnn_confidence = get_prediction(cnn_model, preprocessed_image_cnn, class_labels)
                vgg_class, vgg_confidence = get_prediction(vgg_model, preprocessed_image_vgg, class_labels)
                
                # Display results
                st.subheader("Analysis Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### CNN Model {get_terrain_emoji(cnn_class)}")
                    st.write(f"Terrain Type: **{cnn_class}**")
                    st.write(f"Confidence: **{cnn_confidence:.2%}**")
                    st.progress(cnn_confidence)
                
                with col2:
                    st.markdown(f"### VGG16 Model {get_terrain_emoji(vgg_class)}")
                    st.write(f"Terrain Type: **{vgg_class}**")
                    st.write(f"Confidence: **{vgg_confidence:.2%}**")
                    st.progress(vgg_confidence)
                
                # Show agreement between models
                if cnn_class == vgg_class:
                    st.success(f"✅ Both models identify this as {cnn_class} terrain with high confidence!")
                else:
                    st.warning(f"⚠️ The models have different predictions. This might be a complex terrain with mixed characteristics of {cnn_class} and {vgg_class}.")
                
                # Additional technical details in an expander
                with st.expander("Show Technical Details"):
                    st.markdown("""
                    **Model Information:**
                    - CNN Model: Custom CNN architecture optimized for terrain classification
                    - VGG16 Model: Transfer learning approach using pre-trained VGG16 architecture
                    
                    **Image Processing:**
                    - CNN Input Size: 224x224 pixels
                    - VGG16 Input Size: 150x150 pixels
                    - Normalization: Pixel values scaled to [0,1]
                    """)

    except Exception as e:
        st.error(f"Error loading models or making predictions: {str(e)}")
        st.info("""
        Please ensure:
        1. Both model files are in the same directory as this script
        2. The files are named correctly:
           - terrain_classifier_cnnmodel.h5
           - terrain_analysis_vgg16model.h5
        """)

if __name__ == "__main__":
    main()