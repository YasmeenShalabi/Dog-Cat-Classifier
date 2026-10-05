<h1>Cat vs Dog Image Classifier </h1>

A convolutional neural network (CNN) built with TensorFlow/Keras to classify images as cats or dogs. The model is trained on a dataset of labeled images and uses data augmentation to improve generalization. After training, you can test the model on new images to see whether it predicts cat or dog.


**Dataset**

This project uses the Kaggle Cat and Dog dataset:

https://www.kaggle.com/datasets/tongpython/cat-and-dog/data

Training and validation images are organized into separate directories.


**Model Architecture**

The CNN includes:

4 convolution + max‑pooling blocks

Flatten layer

Dense layer with 512 units

Output layer with sigmoid activation for binary classification

**Features**

Data augmentation (rotation, zoom, flips, shifts)

Training + validation accuracy/loss visualization

Custom function to predict new images

Simple binary output: Cat or Dog

**How to Run**

Update dataset paths in the script.

Test a new image using:

predict_image(model, r"path_to_image.jpeg")

Prediction Example

The script prints:

“The image is predicted to be a Dog…”  

or

“The image is predicted to be a Cat…”
