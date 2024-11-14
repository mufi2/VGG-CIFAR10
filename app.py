

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Define the VGG architecture class
class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers([
            64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 
            512, 512, 512, 'M', 512, 512, 512, 'M'
        ])
        self.fcs = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == int:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU()
                ]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

# CIFAR-10 classes
cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load the model and weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG_net(in_channels=3, num_classes=10).to(device)
model.load_state_dict(torch.load("vgg16_cifar10_state_dict.pth", map_location=device))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Prediction function
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = output.max(1)
        class_name = cifar10_classes[predicted.item()]
    return class_name

# Streamlit app code
st.title("CIFAR-10 Image Classifier")
st.write("Upload an image, and the model will predict its class from CIFAR-10 categories.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Make prediction
    class_name = predict_image(image)
    
    st.write(f"The image is predicted to be: **{class_name}**")
