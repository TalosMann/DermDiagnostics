import streamlit as st
from PIL import Image
import torch
import urllib.request
from fastai.learner import load_learner
from pathlib import Path

# Hugging Face model URL (Replace with your actual model URL)
MODEL_URL = "https://huggingface.co/TalosMann/DermDiagnostics/resolve/main/body_images_resnet50_linux.pkl"
MODEL_FILENAME = "body_images_resnet50_linux.pkl"

# Function to download model if not available locally
@st.cache_resource
def load_model():
    model_path = Path(MODEL_FILENAME)
    
    # Download model if it doesn't exist
    if not model_path.exists():
        with st.spinner("Downloading model..."):
            urllib.request.urlretrieve(MODEL_URL, str(model_path))  # Convert path to string

    # Load the model using fastai's load_learner
    try:
        learner = load_learner(model_path, cpu=True)
        st.success("Model loaded successfully!")
        return learner
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()

# Streamlit UI
st.title("Skin Disease Classification App")
st.write("Upload an image to get the top 3 possible skin conditions.")

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None and model is not None:
    # Open and display image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run prediction
    try:
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
    except Exception as e:
        st.error(f"Error during prediction: {e}")