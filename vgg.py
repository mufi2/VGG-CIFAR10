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

# Function to predict image class
def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = output.max(1)
        class_name = cifar10_classes[predicted.item()]
    
    print(f"The image is predicted to be: {class_name}")

# Specify the image path here
image_path = "archive\cat-facts.jpg"  # Replace with your image path
predict_image(image_path)


import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

transform = transforms.Compose([
    transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.NEAREST),  # Increase size without blurring
    transforms.ToTensor()
])

dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

samples_per_class = 10
class_samples = {class_name: [] for class_name in cifar10_classes}

for img, label in dataset:
    class_name = cifar10_classes[label]
    if len(class_samples[class_name]) < samples_per_class:
        class_samples[class_name].append(img)
    if all(len(samples) == samples_per_class for samples in class_samples.values()):
        break

fig, axes = plt.subplots(len(cifar10_classes), samples_per_class + 1, figsize=(22, 14))
fig.suptitle("CIFAR-10 Classes with 10 Sample Images Each", fontsize=16)

for row, class_name in enumerate(cifar10_classes):
    axes[row, 0].text(0.5, 0.5, class_name, fontsize=12, ha='center', va='center')
    axes[row, 0].axis('off')
    for col in range(samples_per_class):
        img_tensor = class_samples[class_name][col]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * 0.5) + 0.5
        axes[row, col + 1].imshow(img_np)
        axes[row, col + 1].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.9, left=0.1)
plt.show()

