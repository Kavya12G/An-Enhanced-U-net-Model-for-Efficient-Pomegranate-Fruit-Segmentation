import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from keras.applications import DenseNet121  # Import DenseNet121
from keras.models import Model
from keras.callbacks import ModelCheckpoint

# Define data path and class names
data_path = r"C:\Users\kavya\OneDrive\Desktop\dataset\dataIH.csv\diseased_fruits"  # Update to the correct path
class_names = ["Healthy", "Alternaria", "Bacterial Blight", "Cercospora", "Fruit Rot"]

# Load and preprocess images
images = []
labels = []
for class_index, class_name in enumerate(class_names):
    class_folder = os.path.join(data_path, class_name)
    for image_name in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))  # Input size for DenseNet121
        images.append(image)
        labels.append(class_index)

images = np.array(images)
labels = np.array(labels)

# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)

# Load pre-trained DenseNet121 model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Adding custom classification layers on top
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
predictions = layers.Dense(len(class_names), activation='softmax')(x)

# Creating the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define a callback to store the training history
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs={}):
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))

# Create an instance of the callback
accuracy_history = AccuracyHistory()

# Train the model and store history
history = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_data=(x_test, y_test), callbacks=[accuracy_history])

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

# Save the model
save_path = "densenet_model.h5"
try:
    model.save(save_path)
    print("Model saved at:", save_path)
except Exception as e:
    print("Error saving model:", e)
