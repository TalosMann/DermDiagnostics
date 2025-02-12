import streamlit as st
from PIL import Image
import torch
import urllib.request
import os
from fastai.learner import load_learner

# Hugging Face model URL (Replace with your actual model URL)
MODEL_URL = "https://huggingface.co/TalosMann/DermDiagnostics/resolve/main/body_images_resnet50_linux.pkl"

# Function to download model if not available locally
@st.cache_resource
def load_model():
    model_path = "body_images_resnet50.pkl"
    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            urllib.request.urlretrieve(MODEL_URL, model_path)
    return load_learner(model_path)

# Load the model
model = load_model()

# Streamlit UI
st.title("Skin Disease Classification App")
st.write("Upload an image to get the top 3 possible skin conditions.")

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Open and display image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run prediction
    pred_class, pred_idx, outputs = model.predict(image)

    # Get top 3 predictions
    top3_probs, top3_indices = torch.topk(outputs, 3)
    top3_probs = top3_probs.tolist()
    top3_indices = top3_indices.tolist()

    # Display results
    st.write("### Top 3 Predictions:")
    for i in range(3):
        class_name = model.dls.vocab[top3_indices[i]]
        probability = top3_probs[i] * 100
        st.write(f"**{class_name}** - {probability:.2f}% confidence")
