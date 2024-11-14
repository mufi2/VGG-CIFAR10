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

## Acknowledgments

- The CIFAR-10 dataset: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Streamlit](https://streamlit.io/) for the interactive web app.
- VGG-16 architecture as a base for model training.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
