import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import csv

class VGG11(nn.Module):
    def __init__(self, num_classes=36):
        super(VGG11, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load the class mapping from CSV
class_mapping = {}
with open('map.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        class_mapping[row[0]] = row[1]

# Create an instance of your model
model = VGG11()  # Replace with your actual model class instantiation

# Load the trained weights into the model
model.load_state_dict(torch.load('bhanucha_hkongara_assignment2_part4.h5', map_location=torch.device('cpu')))
model.eval()

# Define a function for image preprocessing
def preprocess_image(image):
    # Convert the image to single channel (grayscale)
    image = image.convert('L')
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),  # Grayscale normalization
    ])
    image = transform(image).unsqueeze(0)
    return image

# Set the title and description of the app
st.title("Image Classification with PyTorch")
st.write("Upload an image and let the model classify it.")

# Upload an image
image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if image is not None:
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = Image.open(image).convert('RGB')
    img = preprocess_image(img)

    # Make predictions
    with st.spinner("Classifying..."):
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()
            predicted_label = class_mapping.get(str(predicted_class), "Unknown")

    # Display the predicted result
    st.subheader("Predicted Class:")
    st.write(predicted_label)

# Add a link to the class mapping and model information
st.write("View class mapping: [map.csv](map.csv)")
st.write("Model Information: [bhanucha_hkongara_assignment2_part4.h5](bhanucha_hkongara_assignment2_part4.h5)")
