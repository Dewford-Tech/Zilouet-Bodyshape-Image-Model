import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import torchvision.models as models
import torch.nn as nn

# Define the transformations (same as those used during training/testing)
transform = T.Compose([
    T.Grayscale(num_output_channels=3),  # Convert image to grayscale and replicate channels
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the fine-tuned model
@st.cache_resource
def load_model():
    model = models.efficientnet_b4(pretrained=False)
    num_classes = 5  # Update this to match the number of classes in your trained model
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.BatchNorm1d(model.classifier[1].in_features),
        nn.Linear(model.classifier[1].in_features, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Linear(128, num_classes)
    )
    # Load the model weights onto the CPU
    model.load_state_dict(torch.load('enhance_face.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Class names (update with your actual class names)
class_names = ["Heart", "Oblong", "Oval", "Round", "Square"]  # Update to match the number of classes
 
# Streamlit App
st.title("Face Shape Classification")
st.write("Upload an image to classify its face shape")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Apply transformations
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    
    # Get the predicted class label
    predicted_label = class_names[predicted_class.item()]
    
    st.write(f"Predicted class: **{predicted_label}**")
