from keras.layers import Conv2D, Activation, MaxPool2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
import keras
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """
    This function plots the confusion matrix using seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.show()

# Path and categories
data_path = 'leapGestRecog'
categories = ["01_palm", '02_l', '03_fist', '04_fist_moved', '05_thumb', '06_index', '07_ok', '08_palm_moved', '09_c', '10_down']
imgsz = 50
image_data = []

# Loading and processing images
for i in os.listdir(data_path):
    for category in categories:
        classindex = categories.index(category)
        imgpath = os.path.join(data_path, i, category)
        for img in os.listdir(imgpath):
            img_arr = cv2.imread(os.path.join(imgpath, img), cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(img_arr, (imgsz, imgsz))
            image_data.append([resized_img, classindex])
            print('converting image to greyscale ',img)

# Separating input data and labels
input_data = []
label_data = []

for x, y in image_data:
    input_data.append(x)
    label_data.append(y)

# Converting to numpy arrays
input_data = np.array(input_data)
label_data = np.array(label_data)

# Reshaping input data to include channel dimension (for grayscale images)
input_data = input_data.reshape(-1, imgsz, imgsz, 1)

# One-hot encoding the labels
label = keras.utils.to_categorical(label_data)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_data, label, test_size=0.2, random_state=42, shuffle=True)

# Checking shapes
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Defining the model
model = keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(imgsz, imgsz, 1)),
    MaxPool2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')  # 10 output units for 10 classes
])

# Model summary
model.summary()

# Compiling the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fitting the model with training data
history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_split=0.2
)


def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
plot_training_history(history)

PREDICTION = model.predict(X_test)

# Convert predictions and true labels to class indices
y_pred_classes = np.argmax(PREDICTION, axis=1)  # Predicted class indices
y_test_classes = np.argmax(y_test, axis=1)      # True class indices

# Plot confusion matrix
plot_confusion_matrix(y_test_classes, y_pred_classes, title='Confusion Matrix')

model.save('hand_gesture.h5')