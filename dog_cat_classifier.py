import tensorflow as tf #provides tools for building and training neural networks
from tensorflow.keras.preprocessing.image import ImageDataGenerator #improves models ability to generalize by creating variations in training data
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# https://www.kaggle.com/datasets/tongpython/cat-and-dog/data
# Define paths to the dataset (update these paths with the actual dataset location)
train_dir = r'C:\Users\17325\Downloads\AI_Udemy\archive (5)\training_set\training_set'
validation_dir = r'C:\Users\17325\Downloads\AI_Udemy\archive (5)\test_set\test_set'

# Define ImageDataGenerators for data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,         #Rescale pixel values (0-255) to (0-1)
    rotation_range=40,      #Randomly rotate images
    width_shift_range=0.2,  #Randomly shift images horizontally
    height_shift_range=0.2,  #Randomly shift images vertically
    shear_range=0.2,            #Randomly shear images
    zoom_range=0.2,             #Randomly zoom in on images
    horizontal_flip=True,       #Randomly flip images horizontally
    fill_mode='nearest'         #Fill pixels that may have been lost after transformation
)

#Rescale validation data (no data augmentation needed)
validation_datagen = ImageDataGenerator(rescale=1./255)

#Load training and validation data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),  #Resize all images to 150x150
    batch_size=32,          #Loads 32 images at a time in batches
    class_mode='binary'     #Binary classification (Dog or Cat)
)

validation_generator = validation_datagen.flow_from_directory( #flow_from_dir loads images from directories
    validation_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

#Define the CNN model
model = models.Sequential()

#First convolutional layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3))) #extract spatial features from the images; 32 filters and 3x3pixel size
model.add(layers.MaxPooling2D((2,2))) #reduces spatial dimensions of feature maps by taking max value in 2x2 blocks

#Second convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu')) #64 filters
model.add(layers.MaxPooling2D((2,2)))

#Third convolutional layer
model.add(layers.Conv2D(128, (3, 3), activation='relu')) #128 filters
model.add(layers.MaxPooling2D((2,2)))

#Fourth convolutional layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

#Flatten output from the convolutional layers and add fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu')) #call Dense function which is fully connected layers with 512 units of relu activation
model.add(layers.Dense(1, activation='sigmoid')) #final dense layer is an output layer for binary classification

#Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Print summary of the model
model.summary()

#Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=100,    #Number of batches per epoch
    epochs=20,               #Number of epochs to train
    validation_data=validation_generator,
    validation_steps=50     #Number of batches for validation
)

#Plot training and validation accuracy and loss
acc = history.history['accuracy']  #history.history => dictionary containing lists of metric values per epoch
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss'] #value of loss

epochs = range(len(acc))

plt.figure(figsize=(12,6))
plt.subplot(1,2,1) #for where to place the plot
plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()

#Test the model with a new image
from tensorflow.keras.preprocessing import image
import numpy as np

#function loads and preprocesses a new img then makes a pred
def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(150,150)) #Load image and change its size to 150x150
    img_array = image.img_to_array(img) #convert image to array
    img_array = np.expand_dims(img_array, axis=0) #Add batch dimension
    img_array /= 255.0 #Normalize the image (rescale pixel values to [0,1])

    prediction=model.predict(img_array) #make the prediction
    print("Raw prediction:", prediction)

    if prediction[0] > 0.5:
        print(f"The image is predicted to be a Dog with a confidence of {prediction}")
    else:
        print(f"The image is predicted to be a Cat with a confidence of {1 - prediction}")

#Example: test the classifier with a new image
predict_image(model, r'C:\Users\17325\Downloads\AI_Udemy\test-image.jpeg')
