# Dogs and Wolves Classifier

This project implements a binary classifier to distinguish between dogs and wolves using a custom convolutional neural network (CNN) built with PyTorch.


## Setup

### Prerequisites

- Python 3.7 or higher
- PyTorch
- torchvision
- matplotlib
- numpy
- PIL

### Installation

1. **Clone the repository:**

   git clone https://github.com/yourusername/DogsAndWolvesClassifier.git
   cd DogsAndWolvesClassifier

2. **Create and activate a virtual environment:**

    python3 -m venv venv
    source venv/bin/activate

3. **Install the dependencies:**

    pip3 install torch torchvision matplotlib numpy pillow

## Model training and evaluation

    To train the model, run the following command:

    python3 main.py train --data_path path/to/dataset 
    (Default path is data/)

    Make sure to change directory that specifies where the weights will be saved after training in train.py file (Line 130)

## Predicting Images

    To predict the class of images in a directory, run:

    python3 main.py predict --image_path path/to/your/images/ --weights_path path/to/your/weights.pth
    (Default for --image path is /newImages so you can add new images to this folder)
    (Default for --weights_path is weights/old/final_model_weights.pth)

## Project Components

    models/
	    cnn.py: Contains the definition of the CNN model.
	    early_stopping.py: Implements early stopping to avoid overfitting.

    scripts/

	    train.py: Script for training the model.
	    evaluate.py: Script for evaluating the model.
	    predict.py: Script for predicting the class of images.

## Notes

    The project supports GPU acceleration. If you have a CUDA-compatible GPU, the training and evaluation will automatically use it. Otherwise, it will fall back to CPU

## Authors

    dis012
