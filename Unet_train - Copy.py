import cv2
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
import time
import matplotlib.pyplot as plt

def unet_model(input_shape):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # Decoder
    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = Conv2D(256, 2, activation='relu', padding='same')(up5)
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(128, 2, activation='relu', padding='same')(up6)
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(64, 2, activation='relu', padding='same')(up7)
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def load_and_process_images(image_files, mask_images_dir, image_size):
    images = []
    masks = []
    for img_file in image_files:
        img_path = os.path.join(mask_images_dir, img_file)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, image_size)
                images.append(img)
                
                # Load corresponding mask image
                mask_file = img_file.replace(".jpg", "_mask.jpg")  # Assuming mask images have "_mask" suffix
                mask_path = os.path.join(mask_images_dir, mask_file)
                if not os.path.exists(mask_path):
                    # If mask image not found, try loading the image without the "_mask" suffix
                    mask_file = img_file.replace(".jpg", ".jpg")  
                    mask_path = os.path.join(mask_images_dir, mask_file)
                if os.path.exists(mask_path):
                    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask image in grayscale
                    mask_img = cv2.resize(mask_img, image_size)  # Resize mask image to match original image
                    masks.append(mask_img)
                else:
                    print(f"Mask image not found: {mask_file}")
            else:
                print(f"Error loading image: {img_file}")
        except Exception as e:
            print(f"Error loading image: {img_file}, {e}")
    return np.array(images), np.array(masks)

# Directory paths containing original images and mask images
original_images_dir = r"C:\Users\kavya\OneDrive\Desktop\unet 1\original\f_original"
mask_images_dir = r"C:\Users\kavya\OneDrive\Desktop\unet 1\mask\f_mask"

# List files in the directories
original_image_files = os.listdir(original_images_dir)

# Define image size
image_size = (64, 64)

# Load and process images in batches
batch_size = 32
num_batches = len(original_image_files) // batch_size
X_train_batches = [load_and_process_images(original_image_files[i*batch_size:(i+1)*batch_size], mask_images_dir, image_size) for i in range(num_batches)]
if len(original_image_files) % batch_size != 0:
    X_train_batches.append(load_and_process_images(original_image_files[num_batches*batch_size:], mask_images_dir, image_size))

# Initialize empty list to store processed mask images
Y_train_batches = []

# Concatenate batches
X_train = np.concatenate([x_train_batch for x_train_batch, _ in X_train_batches])
Y_train = np.concatenate([y_train_batch for _, y_train_batch in X_train_batches])

# Check if images were loaded successfully
if len(X_train) == 0 or len(Y_train) == 0:
    print("No images loaded successfully. Exiting...")
    exit()

# Normalize image pixel values to range [0, 1]
X_train = X_train / 255.0
Y_train = Y_train / 255.0

# Define the input shape
input_shape = X_train[0].shape

# Create the U-Net model
model = unet_model(input_shape)

# Define the optimizer with the desired learning rate
optimizer = Adam(learning_rate=0.0001)

# Compile the model with the optimizer
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
start_time = time.time()
history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.1)
end_time = time.time()

# Calculate total training time
total_time = end_time - start_time
print("Total Training Time:", total_time, "seconds")

# Save the trained model in the recommended format
model.save("unet_model.keras")

# Extract overall training accuracy
overall_training_accuracy = history.history['accuracy'][-1]
print("Overall Training Accuracy:", overall_training_accuracy)

# Calculate Dice coefficient
# Assuming Y_train is the ground truth mask and model predictions are stored in 'predictions'
predictions = model.predict(X_train)
dice_coefficient = []
for i in range(len(Y_train)):
    intersection = np.sum(Y_train[i] * predictions[i])
    union = np.sum(Y_train[i]) + np.sum(predictions[i])
    dice_coefficient.append(2 * intersection / (union + 1e-8))  # Adding 1e-8 to avoid division by zero
print("Dice Coefficient:", np.mean(dice_coefficient))

# Plot training and validation accuracy
# (assuming the history object contains 'accuracy' and 'val_accuracy' keys)
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(training_accuracy, label='Training Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()



