# main.py

import streamlit as st
import os
import io
import base64
import json
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import requests
from PIL import Image
from gtts import gTTS
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from openai import OpenAI
from dotenv import load_dotenv

# ------------------ Load Environment ------------------
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENWEATHER_API_KEY or not OPENAI_API_KEY:
    st.error("API keys missing. Add them to .env file.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Plant Disease Recognition", layout="wide")

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "trained_plant_disease_model01.keras")
    if not os.path.exists(model_path):
        st.error(f"Model not found: {model_path}")
        st.stop()
    return tf.keras.models.load_model(model_path)

model = load_model()


# ------------------ Prediction ------------------
def model_prediction(image_file):
    img = tf.keras.preprocessing.image.load_img(image_file, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    return prediction


# ------------------ Class Labels ------------------
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
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]


def clean_label(label):
    return label.replace("___", " ").replace("_", " ").title()


# ------------------ Layer 1: Vision API Leaf Validator ------------------
def validate_leaf_with_vision(image_bytes: bytes) -> tuple:
    """
    Uses GPT-4o vision to check if image is a plant leaf.
    Returns (is_leaf: bool, reason: str, api_success: bool)
    """
    try:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict plant leaf image validator. "
                        "Look at the image carefully and determine if it shows a plant leaf or leaves. "
                        "Only respond with a JSON object in this exact format with no extra text: "
                        '{"is_leaf": true or false, "reason": "one short sentence"}'
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                                "detail": "low"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Is this a plant leaf image? Reply ONLY with the JSON."
                        }
                    ]
                }
            ],
            max_tokens=100
        )

        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        return bool(result.get("is_leaf", False)), result.get("reason", "Unknown"), True

    except Exception as e:
        # API failed — signal to use fallback
        return False, str(e), False


# ------------------ Layer 2: Pixel-based Leaf Validator (Fallback) ------------------
def validate_leaf_with_pixels(image_bytes: bytes) -> tuple:
    """
    Fallback validator using green/brown/yellow pixel ratio analysis.
    Plant leaves have significant green channel dominance.
    Returns (is_leaf: bool, reason: str)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((100, 100))
        pixels = np.array(img, dtype=float)

        R = pixels[:, :, 0]
        G = pixels[:, :, 1]
        B = pixels[:, :, 2]

        # Green dominance (healthy leaf)
        green_dominant = (G > R) & (G > B)
        green_ratio = np.sum(green_dominant) / (100 * 100)

        # Yellow tones (stressed/diseased leaf)
        yellow_dominant = (R > 100) & (G > 100) & (B < 100) & (G >= R * 0.7)
        yellow_ratio = np.sum(yellow_dominant) / (100 * 100)

        # Brown tones (diseased/dry leaf)
        brown_dominant = (R > 80) & (G > 40) & (G < R * 0.85) & (B < 80)
        brown_ratio = np.sum(brown_dominant) / (100 * 100)

        leaf_color_ratio = green_ratio + (yellow_ratio * 0.5) + (brown_ratio * 0.3)

        if leaf_color_ratio >= 0.15:
            return True, f"Image contains {leaf_color_ratio*100:.0f}% leaf-like colors"
        else:
            return False, f"Only {leaf_color_ratio*100:.0f}% leaf-like colors detected — does not appear to be a plant leaf"

    except Exception as e:
        return True, f"Pixel check failed: {e}"  # fail open to avoid blocking real users


# ------------------ Combined Validator ------------------
def is_plant_leaf(image_bytes: bytes) -> tuple:
    """
    Tries GPT-4o vision first. Falls back to pixel color analysis if API fails.
    Returns (is_leaf: bool, reason: str, method: str)
    """
    is_leaf, reason, api_success = validate_leaf_with_vision(image_bytes)

    if api_success:
        return is_leaf, reason, "AI Vision"
    else:
        # Vision API failed — use pixel fallback
        is_leaf, reason = validate_leaf_with_pixels(image_bytes)
        return is_leaf, reason, "Color Analysis (fallback)"


# ------------------ Weather ------------------
def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        r = requests.get(url).json()
        temp = r["main"]["temp"]
        cond = r["weather"][0]["description"]
        return f"{temp}°C , {cond}"
    except:
        return "Weather not found"


# ------------------ Text to Speech ------------------
def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()


# ------------------ PDF Generator ------------------
def generate_pdf(disease, text):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height - 60, f"Disease: {disease}")
    c.setFont("Helvetica", 12)
    text_obj = c.beginText(40, height - 100)
    for line in text.split("."):
        text_obj.textLine(line)
    c.drawText(text_obj)
    c.save()
    buffer.seek(0)
    return buffer.read()


# ------------------ AI Solution ------------------
def ai_solution(disease):
    prompt = f"""
    Explain plant disease {disease}.
    
    1 Symptoms
    2 Causes
    3 Treatment and prevention
    
    Use simple language for farmers.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are agriculture expert"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


# ------------------ Chatbot ------------------
def agri_chat(question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are helpful Indian agriculture assistant"},
            {"role": "user", "content": question}
        ],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


# ------------------ Sidebar ------------------
st.sidebar.title("🌿 Smart Plant Assistant")

mode = st.sidebar.radio(
    "Navigation",
    ["Home", "About", "Disease Recognition", "Agri Chatbot"]
)

city = st.sidebar.text_input("Enter City", "Pune")
st.sidebar.info(get_weather(city))

if "history" not in st.session_state:
    st.session_state.history = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ------------------ Home ------------------
if mode == "Home":
    st.title("🌱 Plant Disease Recognition System")
    st.write("Upload a plant leaf image to detect disease")
    if os.path.exists("home_page.jpeg"):
        st.image("home_page.jpeg", use_column_width=True)


# ------------------ About ------------------
elif mode == "About":
    st.header("About")
    st.info("""
    CNN based Plant Disease Detection system.
    
    Uses PlantVillage dataset.
    
    AI provides disease solution and farming chatbot.
    """)
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history))


# ------------------ Disease Recognition ------------------
elif mode == "Disease Recognition":
    st.header("Detect Plant Disease")

    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])
    camera_img = st.camera_input("Or Capture Image")

    img = uploaded_file or camera_img

    if img:
        st.image(img, width=300)

        if st.button("Predict"):

            # ---- Step 1: Validate it's a leaf ----
            with st.spinner("🔍 Checking if image is a plant leaf..."):
                image_bytes = img.getvalue()
                leaf_valid, reason, method = is_plant_leaf(image_bytes)

            if not leaf_valid:
                st.error("❌ This does not appear to be a plant leaf image!")
                st.warning(f"**Reason:** {reason}")
                st.info("💡 Please upload a clear photo of a plant leaf (tomato, potato, apple, etc.)")

            else:
                st.success(f"✅ Leaf confirmed ({method})")

                # ---- Step 2: Run disease prediction ----
                with st.spinner("🧪 Analyzing leaf for disease..."):
                    img.seek(0)
                    preds = model_prediction(img)

                idx = np.argmax(preds)
                disease = clean_label(CLASS_NAMES[idx])
                confidence = preds[idx] * 100

                if confidence < 60:
                    st.warning(f"⚠️ Low confidence ({confidence:.2f}%) — try a clearer, closer leaf photo.")
                else:
                    st.success(f"🌿 Disease: **{disease}**")

                st.info(f"Confidence: {confidence:.2f}%")

                st.session_state.last_prediction = disease
                st.session_state.history.append({
                    "Time": str(datetime.now()),
                    "Prediction": disease,
                    "Confidence": f"{confidence:.2f}%"
                })

    if "last_prediction" in st.session_state:
        if st.button("Get AI Solution"):
            solution = ai_solution(st.session_state.last_prediction)
            st.markdown(solution)

            audio = text_to_speech(solution)
            st.audio(audio)

            pdf = generate_pdf(st.session_state.last_prediction, solution)
            st.download_button(
                "Download PDF",
                pdf,
                file_name="disease_solution.pdf"
            )


# ------------------ Chatbot ------------------
elif mode == "Agri Chatbot":
    st.header("Agriculture Chatbot")
    question = st.text_input("Ask farming question")
    if question:
        answer = agri_chat(question)
        st.write(answer)