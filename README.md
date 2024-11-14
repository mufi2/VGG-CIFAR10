# VGG-CIFAR10
# CIFAR-10 Image Classifier

This project is a web application that classifies images into one of the CIFAR-10 classes. The app is built with [Streamlit](https://streamlit.io/) and uses a pre-trained VGG-16 model for CIFAR-10 classification.

## Project Structure

- **app.py**: The main Streamlit app where users can upload an image, and the model predicts the class.
- **vgg16_cifar10_state_dict.pth**: The saved model weights for VGG-16 trained on CIFAR-10 (not included, please add your model file).

## Model Architecture

The VGG-16 model architecture is defined in the `VGG_net` class. The architecture includes convolutional layers, followed by fully connected layers, with ReLU activations and dropout for regularization.

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

![Example](example-output.png)

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
