import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import json
import urllib.request

# =====================================================================
# 🧠 DEEP LEARNING ENGINE (Pre-trained Neural Network)
# =====================================================================

@st.cache_resource
def load_deep_learning_model():
    """Requirement 1: Load a pre-trained Model using PyTorch"""
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.eval()  # Set model to evaluation mode
    
    # Download the standard 1,000 class names mapping
    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    with urllib.request.urlopen(labels_url) as url:
        classes = [line.decode("utf-8").strip() for line in url.readlines()]
        
    return model, classes, weights

neural_network, class_labels, model_weights = load_deep_learning_model()

def process_and_classify_image(uploaded_img):
    """Requirement 2: Perform recognition on sample input"""
    img = Image.open(uploaded_img).convert("RGB")
    
    # Define standard image preprocessing mutations required by MobileNetV2
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
    
    # Run the image tensor through the layers
    with torch.no_grad():
        output = neural_network(input_batch)
        
    # Convert raw outputs into softmax probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Extract the top 3 high-confidence indices
    top3_prob, top3_catid = torch.topk(probabilities, 3)
    
    results = []
    for i in range(top3_prob.size(0)):
        results.append((class_labels[top3_catid[i]], top3_prob[i].item()))
    return results

# =====================================================================
# 🎨 UI DASHBOARD LAYER (Streamlit Frontend)
# =====================================================================

st.set_page_config(page_title="AI Vision Engine", page_icon="👁️", layout="centered")

st.title("👁️ Project 4: Image Recognition Engine")
st.caption("Internship Portfolio Milestone - Deep Learning Inference via PyTorch")
st.write("---")

st.subheader("📸 Step 1: Upload a Sample Image")
st.write("Upload any photo (e.g., a dog, banana, car) to test the neural network model:")

image_file = st.file_uploader(
    "Choose an image file...", type=["jpg", "jpeg", "png", "webp"]
)

if image_file is not None:
    st.write("---")
    col_view, col_predictions = st.columns([1, 1], gap="large")
    
    with col_view:
        st.subheader("🖼️ Ingested Input Image")
        st.image(image_file, use_container_width=True)
        
    with col_predictions:
        st.subheader("🤖 Requirement 3: Display Model Outputs")
        st.write("The deep learning network processed the input pixels and extracted these feature predictions:")
        
        with st.spinner("Analyzing image features..."):
            top_predictions = process_and_classify_image(image_file)
            
        for index, (label, confidence) in enumerate(top_predictions):
            percentage_score = confidence * 100
            clean_label = label.replace("_", " ").title()
            
            if index == 0:
                with st.container(border=True):
                    st.markdown(f"🏆 **Primary Prediction: {clean_label}**")
                    st.metric(label="Match Confidence", value=f"{percentage_score:.2f}%")
                    st.progress(percentage_score / 100)
            else:
                st.write(f"🔹 Alternative Guess: **{clean_label}** ({percentage_score:.1f}%)")
else:
    st.info("💡 Please drag and drop or upload an image above to kick off the AI vision tracking logic loop!")