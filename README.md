# VGG-CIFAR10
# CIFAR-10 Image Classifier

![1_CIFAR-10   VGG16 for Image Detection](https://github.com/user-attachments/assets/e2711c61-9221-4dfa-b474-ae3e7ff8d3f6)

This project is a web application that classifies images into one of the CIFAR-10 classes. The app is built with [Streamlit](https://streamlit.io/) and uses a pre-trained VGG-16 model for CIFAR-10 classification.

## Project Structure

- **app.py**: The main Streamlit app where users can upload an image, and the model predicts the class.
- **vgg16_cifar10_state_dict.pth**: The saved model weights for VGG-16 trained on CIFAR-10 (not included, please add your model file).

## Model Architecture

The VGG-16 model architecture is defined in the `VGG_net` class. The architecture includes convolutional layers, followed by fully connected layers, with ReLU activations and dropout for regularization.
![image](https://github.com/user-attachments/assets/95533ec3-09d8-4ef0-b892-90238aa31466)

| Layer Type               | Configuration                                                                                                  | Output Shape       | Description                                                                                               |
|--------------------------|-----------------------------------------------------------------------------------------------------------------|--------------------|-----------------------------------------------------------------------------------------------------------|
| Input Layer              | -                                                                                                               | 3 x 32 x 32       | CIFAR-10 input image (3 channels for RGB, 32x32 resolution)                                               |
| Convolution + ReLU       | Conv2D(3, 64, kernel_size=3, padding=1) + BatchNorm2d + ReLU                                                   | 64 x 32 x 32      | First 3x3 convolution layer with 64 filters, padding for same output size, followed by ReLU activation.   |
| Convolution + ReLU       | Conv2D(64, 64, kernel_size=3, padding=1) + BatchNorm2d + ReLU                                                  | 64 x 32 x 32      | Another 3x3 convolution layer with 64 filters, followed by ReLU.                                          |
| Max Pooling              | MaxPool2d(kernel_size=2, stride=2)                                                                             | 64 x 16 x 16      | 2x2 max-pooling reduces spatial dimensions by half.                                                       |
| Convolution + ReLU       | Conv2D(64, 128, kernel_size=3, padding=1) + BatchNorm2d + ReLU                                                 | 128 x 16 x 16     | 3x3 convolution layer with 128 filters, followed by ReLU.                                                 |
| Convolution + ReLU       | Conv2D(128, 128, kernel_size=3, padding=1) + BatchNorm2d + ReLU                                                | 128 x 16 x 16     | Another 3x3 convolution layer with 128 filters, followed by ReLU.                                         |
| Max Pooling              | MaxPool2d(kernel_size=2, stride=2)                                                                             | 128 x 8 x 8       | 2x2 max-pooling reduces spatial dimensions by half.                                                       |
| Convolution + ReLU       | Conv2D(128, 256, kernel_size=3, padding=1) + BatchNorm2d + ReLU                                                | 256 x 8 x 8       | 3x3 convolution with 256 filters, followed by ReLU.                                                       |
| Convolution + ReLU       | Conv2D(256, 256, kernel_size=3, padding=1) + BatchNorm2d + ReLU                                                | 256 x 8 x 8       | Another 3x3 convolution with 256 filters.                                                                 |
| Convolution + ReLU       | Conv2D(256, 256, kernel_size=3, padding=1) + BatchNorm2d + ReLU                                                | 256 x 8 x 8       | Additional 3x3 convolution with 256 filters, followed by ReLU.                                            |
| Max Pooling              | MaxPool2d(kernel_size=2, stride=2)                                                                             | 256 x 4 x 4       | 2x2 max-pooling reduces spatial dimensions by half.                                                       |
| Convolution + ReLU       | Conv2D(256, 512, kernel_size=3, padding=1) + BatchNorm2d + ReLU                                                | 512 x 4 x 4       | 3x3 convolution with 512 filters.                                                                         |
| Convolution + ReLU       | Conv2D(512, 512, kernel_size=3, padding=1) + BatchNorm2d + ReLU                                                | 512 x 4 x 4       | Another 3x3 convolution with 512 filters.                                                                 |
| Convolution + ReLU       | Conv2D(512, 512, kernel_size=3, padding=1) + BatchNorm2d + ReLU                                                | 512 x 4 x 4       | Additional 3x3 convolution with 512 filters.                                                              |
| Max Pooling              | MaxPool2d(kernel_size=2, stride=2)                                                                             | 512 x 2 x 2       | 2x2 max-pooling reduces spatial dimensions by half.                                                       |
| Convolution + ReLU       | Conv2D(512, 512, kernel_size=3, padding=1) + BatchNorm2d + ReLU                                                | 512 x 2 x 2       | Final block: 3x3 convolution with 512 filters.                                                            |
| Convolution + ReLU       | Conv2D(512, 512, kernel_size=3, padding=1) + BatchNorm2d + ReLU                                                | 512 x 2 x 2       | Another 3x3 convolution layer with 512 filters.                                                           |
| Convolution + ReLU       | Conv2D(512, 512, kernel_size=3, padding=1) + BatchNorm2d + ReLU                                                | 512 x 2 x 2       | Additional 3x3 convolution with 512 filters.                                                              |
| Max Pooling              | MaxPool2d(kernel_size=2, stride=2)                                                                             | 512 x 1 x 1       | Final 2x2 max-pooling layer.                                                                              |
| Flatten                  | -                                                                                                               | 512               | Flatten output for fully connected layers.                                                                |
| Fully Connected + ReLU   | Linear(512, 4096) + ReLU + Dropout(p=0.5)                                                                      | 4096              | First fully connected layer with dropout for regularization.                                              |
| Fully Connected + ReLU   | Linear(4096, 4096) + ReLU + Dropout(p=0.5)                                                                     | 4096              | Second fully connected layer with ReLU and dropout.                                                       |
| Output Layer             | Linear(4096, num_classes)                                                                                      | 10                | Output layer for CIFAR-10 classification, producing 10 class scores.                                      |


## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/cifar10-image-classifier.git
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Download the `vgg16_cifar10_state_dict.pth` model file and place it in the project directory.

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
2. Upload an image (jpg, jpeg, png format) to the app, and it will display the predicted class label.

## Example Output

![image](https://github.com/user-attachments/assets/fbb1adff-63a9-42d6-8b0e-60bc125a6c0f)

## CIFAR-10 Classes

The model classifies images into the following classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck
## Model Performance
| **Metric**          | **Training** | **Testing** |
|---------------------|--------------|-------------|
| **Accuracy**        | 0.9655       | 0.8439      |
| **Precision**       | 0.9663       | 0.8464      |
| **F1 Score**        | 0.9654       | 0.8431      |

### Loss vs Epoch
![image](https://github.com/user-attachments/assets/dbf148c5-69f3-4a44-ba3a-4fa944f50c20)
### Confusion Matrix
![image](https://github.com/user-attachments/assets/359ef637-334d-4282-9302-93e8892f0f9a)

##  Streamlit Web Application
An interactive [Streamlit web application](https://vgg-cifar10-q3gm3daeriaha9rztetykm.streamlit.app/) enables users to upload images of Objects and get predictions in real time.

### 💻 App Interface
The app features a sleek and intuitive interface:
![image](https://github.com/user-attachments/assets/dff9ff65-d920-4f9d-9cf1-791f2ffec7d8)

Upload an image, and the model will instantly classify with high precision.


## Acknowledgments

- VGG paper: [VGG Paper](https://arxiv.org/abs/1409.1556)
- The CIFAR-10 dataset: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Streamlit](https://vgg-cifar10-q3gm3daeriaha9rztetykm.streamlit.app/) for the interactive web app.
- VGG-16 architecture as a base for model training.
  

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
