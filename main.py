# app.py
# Requirements: streamlit, tensorflow, numpy, pandas, matplotlib, requests, gTTS, reportlab, openai, python-dotenv

import streamlit as st
import os
import io
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import requests
from gtts import gTTS
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from openai import OpenAI
from dotenv import load_dotenv  # ✅ For loading API keys securely

# -------------------- Load Environment Variables --------------------
load_dotenv()  # loads .env file from project root

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENWEATHER_API_KEY or not OPENAI_API_KEY:
    st.error("❌ API keys are missing. Please create a .env file with OPENWEATHER_API_KEY and OPENAI_API_KEY.")
    st.stop()

# -------------------- OpenAI Client --------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------- Page Config --------------------
st.set_page_config(page_title="Plant Disease Recognition", layout="wide")

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.keras")

model = load_model()

# -------------------- Prediction --------------------
def model_prediction(image_file):
    img = tf.keras.preprocessing.image.load_img(image_file, target_size=(128, 128))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)[0]
    return preds

# -------------------- Class Names --------------------
CLASS_NAMES = [
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

def clean_label(label):
    return label.replace("___", " ").replace("_", " ").title()

# -------------------- Weather --------------------
def get_weather_for_city(city_name):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={OPENWEATHER_API_KEY}&units=metric"
        r = requests.get(url, timeout=6)
        data = r.json()
        if r.status_code != 200 or "main" not in data:
            return f"Weather not found for '{city_name}'"
        temp = data["main"]["temp"]
        cond = data["weather"][0]["description"].capitalize()
        return f"{temp}°C, {cond}"
    except Exception as e:
        return f"Error fetching weather: {e}"

# -------------------- TTS --------------------
def tts_gtts(text):
    try:
        tts = gTTS(text=text, lang="en")
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except:
        return None

# -------------------- PDF Generator --------------------
def generate_pdf_bytes(disease_name, content):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, height - 60, f"Disease: {disease_name}")
    c.setFont("Helvetica", 12)
    text = c.beginText(40, height - 100)
    for line in content.split(". "):
        text.textLine(line)
    c.drawText(text)
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()

# -------------------- AI Disease Solution --------------------
def ai_generate_solution(disease_name):
    prompt = f"""
    You are an agricultural expert. Explain the plant disease '{disease_name}' in 3 short parts:
    1. What it is and symptoms
    2. Causes
    3. Treatment & prevention (natural + modern methods)
    Use simple English.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful agricultural assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=350,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating AI response: {e}"

# -------------------- AI ChatBot --------------------
def agri_chatbot_response(question):
    prompt = f"You are a smart Indian agriculture assistant. Answer this farmer’s question in simple English:\n\nQ: {question}\nA:"
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in Indian agriculture, helping farmers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# -------------------- Streamlit UI --------------------
st.sidebar.title("🌿 Smart Plant Assistant")

app_mode = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "ℹ️ About", "🔬 Disease Recognition", "🤖 Smart Agri ChatBot"]
)

city = st.sidebar.text_input("Enter city for weather", "Pune")
if city:
    st.sidebar.info(get_weather_for_city(city))

if "history" not in st.session_state:
    st.session_state.history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------- HOME --------------------
if app_mode == "🏠 Home":
    st.title("🌱 Plant Disease Recognition System")
    st.write("Upload or capture a leaf image to identify diseases and get AI-based treatment suggestions.")
    if os.path.exists("home_page.jpeg"):
        st.image("home_page.jpeg", use_column_width=True)

# -------------------- ABOUT --------------------
elif app_mode == "ℹ️ About":
    st.header("About the Project")
    st.info("""
    🌿 **Plant Disease Recognition System** uses a Convolutional Neural Network (CNN)
    trained on the PlantVillage dataset to identify crop diseases.
    
    💡 Integrated with OpenAI to provide smart disease treatments and a farming chatbot.
    """)
    if st.sidebar.checkbox("📜 Show Prediction History"):
        if st.session_state.history:
            st.dataframe(pd.DataFrame(st.session_state.history))
        else:
            st.info("No predictions yet.")

# -------------------- DISEASE RECOGNITION --------------------
elif app_mode == "🔬 Disease Recognition":
    st.header("🔍 Detect Plant Disease")

    col1, col2 = st.columns([1.5, 1])
    with col1:
        uploaded_file = st.file_uploader("📸 Upload a Leaf Image", type=["jpg", "jpeg", "png"])
    with col2:
        camera_image = st.camera_input("Or Capture from Camera")

    image_source = uploaded_file or camera_image

    if image_source:
        st.image(image_source, caption="Input Image", use_column_width=True)

        if st.button("🚀 Predict Disease", use_container_width=True):
            with st.spinner("Analyzing image..."):
                preds = model_prediction(image_source)
            top_idx = np.argmax(preds)
            clean_name = clean_label(CLASS_NAMES[top_idx])
            confidence = preds[top_idx] * 100
            st.success(f"🌱 Disease: {clean_name}")
            st.info(f"🎯 Confidence: {confidence:.2f}%")
            st.session_state.last_prediction = clean_name
            st.session_state.history.append({
                "Time": str(datetime.now()),
                "Prediction": clean_name,
                "Confidence": f"{confidence:.2f}%"
            })
            st.balloons()

        if "last_prediction" in st.session_state:
            st.write("---")
            if st.button("💡 View Solution", use_container_width=True):
                with st.spinner("Generating AI-based explanation..."):
                    ai_response = ai_generate_solution(st.session_state.last_prediction)
                st.subheader("🧠 AI Disease Insights")
                st.markdown(ai_response)
                audio_data = tts_gtts(ai_response)
                if audio_data:
                    st.audio(audio_data, format="audio/mp3")
                pdf_bytes = generate_pdf_bytes(st.session_state.last_prediction, ai_response)
                st.download_button(
                    "📥 Download Solution as PDF",
                    pdf_bytes,
                    file_name=f"{st.session_state.last_prediction}_solution.pdf"
                )

        if st.button("🧹 Clear", use_container_width=True):
            st.session_state.clear()
            st.experimental_rerun()
    else:
        st.warning("Please upload or capture an image to proceed.")

# -------------------- CHATBOT --------------------
elif app_mode == "🤖 Smart Agri ChatBot":
    st.header("🤖 Smart Agriculture ChatBot")
    st.write("Ask any question about farming, crop care, fertilizers, irrigation, or weather 🌾")

    # Display previous conversation
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f"**You:** {chat['content']}")
        else:
            st.markdown(f"**AI:** {chat['content']}")

    # Input box for new question
    user_question = st.text_input("Type your question here and press Enter")
    if user_question:
        # Append user question to history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        # Generate AI response
        with st.spinner("Thinking..."):
            response = agri_chatbot_response(user_question)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        # Display response
        st.markdown(f"**AI:** {response}")
