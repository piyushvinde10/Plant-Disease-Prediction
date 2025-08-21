import streamlit as st
import tensorflow as tf
import numpy as np

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Sidebar
st.sidebar.title("🌱 Plant Disease Dashboard")
app_mode = st.sidebar.radio("Navigate", ["🏠 Home", "ℹ️ About", "🔬 Disease Recognition"])

# Main Page
if app_mode == "🏠 Home":
    st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown("""
    <div style='background-color: #e6ffe6; padding: 10px; border-radius: 10px'>
    <h3>Welcome!</h3>
    <p>
    Our mission is to help in identifying plant diseases efficiently.<br>
    <b>Upload an image</b> of a plant, and our system will analyze it to detect any signs of diseases.<br>
    <b>Together, let's protect our crops and ensure a healthier harvest!</b>
    </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant.
    2. **Analysis:** Our system will process the image using advanced algorithms.
    3. **Results:** View the results and recommendations.

    ### Why Choose Us?
    - 🧠 **Accuracy:** State-of-the-art machine learning techniques.
    - 😃 **User-Friendly:** Simple and intuitive interface.
    - ⚡ **Fast and Efficient:** Results in seconds.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar!
    """)

elif app_mode == "ℹ️ About":
    st.header("About")
    st.info("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset.
    It consists of about 87K RGB images of healthy and diseased crop leaves, categorized into 38 classes.
    - **Train:** 70,295 images
    - **Test:** 33 images
    - **Validation:** 17,572 images
    """)

elif app_mode == "🔬 Disease Recognition":
    st.header("🔬 Disease Recognition")
    st.markdown("#### Upload a plant leaf image to detect disease")
    uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.markdown("Click **Predict** to analyze the image.")
        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                result_index = model_prediction(uploaded_file)
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            st.success(f"🌱 Model Prediction: **{class_name[result_index]}**")
            st.balloons()
    else:
        st.warning("Please upload an image to proceed.")
