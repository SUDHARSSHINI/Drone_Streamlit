# Import necessary libraries
import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Create a QnA pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Streamlit App
st.title("Land Irrigation Recommendation System")

# Sidebar to upload or provide image URL
image_source = st.sidebar.radio("Choose image source:", ('Upload Image', 'URL Image'))

# Load image based on user selection
if image_source == 'Upload Image':
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
else:
    image_url = st.sidebar.text_input("Enter image URL")
    if image_url:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

# Display the image
if 'image' in locals():
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Define question input
st.subheader("Ask a Question about the Image")
question = st.text_input("Enter your question", "Does the land need irrigation?")

# Answer the question
if question and 'image' in locals():
    # Use a fixed context for now, but this could be extended in the future
    context = "Need of Irrigation or Not"
    result = qa_pipeline(question=question, context=context)

    # Display result
    st.write(f"Answer: **{result['answer']}**")

    # Function to determine if water is needed
    def needs_water(answer):
        if "water" in answer.lower():
            return "The land needs water."
        else:
            return "The land does not need water."

    # Print if water is needed based on the model's answer
    water_message = needs_water(result['answer'])
    st.write(water_message)
    
    # Display the image with a title showing the answer
    plt.imshow(image)
    plt.axis('off')
    plt.title(f" {result['answer']}")
    st.pyplot(plt)
