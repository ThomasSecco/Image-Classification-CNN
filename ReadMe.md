# Fashion MNIST Image Classification with CNNs 👕👟

## Overview 📊

This project demonstrates the use of **Convolutional Neural Networks (CNNs)** to classify images in the **Fashion MNIST dataset**. The dataset contains grayscale images of 10 categories of clothing and accessories, such as **T-shirts**, **trousers**, and **sneakers**. The goal is to build, train, and evaluate a CNN model that can accurately classify these images. 🎯

## Dataset 📂

The Fashion MNIST dataset contains:
- **60,000 training images**
- **10,000 test images**
- **10 classes**:
  - 👕 T-shirt/top
  - 👖 Trouser
  - 🧥 Pullover
  - 👗 Dress
  - 🧥 Coat
  - 🥿 Sandal
  - 👔 Shirt
  - 👟 Sneaker
  - 👜 Bag
  - 🥾 Ankle boot

Each image is a **28x28 grayscale image**.

## Notebooks 📓

### 1. EDA (Exploratory Data Analysis) 🔍
- Visualize dataset distribution.
- Display sample images for each class.
- Analyze pixel intensity distributions.
- Normalize data for improved training performance.

### 2. Baseline CNN Model 🧠
- Train a simple CNN on the Fashion MNIST dataset.
- Evaluate the model's accuracy and loss on the test set.
- Generate a confusion matrix and classification report.

## Scripts 🖥️

### 1. Data Loader (`src/data_loader.py`)
- Loads the Fashion MNIST dataset.
- Normalizes pixel values.
- Reshapes images to be compatible with CNN input.

### 2. Model Definition (`src/model.py`) 🏗️
- Defines a CNN architecture:
  - Convolutional layers
  - MaxPooling layers
  - Fully connected layers
  - Dropout for regularization

### 3. Training Script (`src/train.py`) 🎯
- Trains the CNN model using the training set.
- Saves the trained model for evaluation.

### 4. Evaluation Script (`src/evaluate.py`) 📊
- Evaluates the model on the test set.
- Generates a confusion matrix and classification report.

## How to Run 🚀

### 1. Clone the Repository
```bash
git clone https://github.com/ThomasSecco/Image-Classification-CNN.git
cd Image-Classification-CNN

```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Notebooks
Navigate to the notebooks/ directory and execute:

- EDA.ipynb to explore the dataset.
- Baseline_Model.ipynb to train and evaluate the CNN.

### 4. Run Training and Evaluation Scripts
```bash
python src/train.py
python src/evaluate.py
```

## Future Work
- Experiment with deeper CNN architectures (e.g., ResNet, VGG).
- Implement data augmentation to improve model generalization.
- Fine-tune hyperparameters using techniques like grid search.

## Contributing 🤝
Feel free to contribute to this project by:

- Suggesting improvements.
- Submitting pull requests.

  ## 🎉 Happy Coding!
