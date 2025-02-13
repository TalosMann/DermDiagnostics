import streamlit as st
from PIL import Image
import torch
import urllib.request
import os
from fastai.learner import load_learner
from pathlib import Path, PosixPath
from pathlib import Path, PosixPath

# Hugging Face model URL (Replace with your actual model URL)
MODEL_URL = "https://huggingface.co/TalosMann/DermDiagnostics/resolve/main/body_images_resnet50_linux.pkl"

# Function to download model if not available locally
@st.cache_resource
def load_model():
    model_path = Path("body_images_resnet50_linux.pkl")  # Use Path() to ensure compatibility
    
    # Download model if it doesn't exist
    if not model_path.exists():
        with st.spinner("Downloading model..."):
            urllib.request.urlretrieve(MODEL_URL, str(model_path))  # Convert path to string

    # Ensure FastAI uses PosixPath
    learner = load_learner(PosixPath(model_path), cpu=True)
    learner.path = PosixPath(learner.path)  # Ensure internal path is PosixPath

    return learner

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
