import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoFeatureExtractor, AutoModelForImageClassification
import torch
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Load models and tokenizers
@st.cache_resource
def load_text_model():
    tokenizer = AutoTokenizer.from_pretrained("climatebert/distilroberta-base-climate-detector")
    model = AutoModelForSequenceClassification.from_pretrained("climatebert/distilroberta-base-climate-detector")
    return tokenizer, model

@st.cache_resource
def load_image_model():
    model_name = "google/vit-base-patch16-224"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return model, feature_extractor

# Classify Climate Text
def classify_climate_text(text):
    tokenizer, model = load_text_model()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=1).item()
    return predicted_class, probs

# Load an image from URL or local file
def load_image(image_path):
    if image_path.startswith('http'):
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_path)
    return img

# Image Classification
def classify_image(image):
    model, feature_extractor = load_image_model()
    pipe = pipeline("image-classification", model=model, feature_extractor=feature_extractor)
    results = pipe(image)
    return results

# Streamlit UI
st.title("Climate Text and Image Classifier")

# Text Classification Section
st.header("Climate Text Classification")
input_text = st.text_area("Enter text related to climate change:")
if st.button("Classify Text"):
    if input_text:
        predicted_class, probs = classify_climate_text(input_text)
        st.write(f"Predicted class: {predicted_class}")
        st.write(f"Class probabilities: {probs}")
    else:
        st.write("Please enter some text.")

# Image Classification Section
st.header("Weather Image Classification")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if st.button("Classify Image"):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        results = classify_image(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        for result in results:
            st.write(f"Label: {result['label']}, Score: {result['score']:.2f}")
    else:
        st.write("Please upload an image.")
