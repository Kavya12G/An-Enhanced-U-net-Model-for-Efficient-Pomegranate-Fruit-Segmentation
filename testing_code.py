# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt

# # Load the trained model
# model = load_model(r"C:\Users\kavya\OneDrive\Desktop\unet 1\unet_model33.h5")

# # Function to preprocess a single test image
# def preprocess_test_image(image_path, image_size):
#     try:
#         img = cv2.imread(image_path)
#         if img is not None:
#             img = cv2.resize(img, image_size)
#             img = np.expand_dims(img, axis=0)  # Add batch dimension
#             return img
#         else:
#             print("Error: Image not loaded successfully.")
#             return None
#     except Exception as e:
#         print(f"Error: {e}")
#         return None

# # Function to calculate test accuracy
# def calculate_test_accuracy(predicted_mask, ground_truth_mask):
#     # Threshold the predicted mask
#     predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8)

#     # Calculate accuracy
#     accuracy = np.mean((predicted_mask_binary == ground_truth_mask).astype(np.float32))
#     return accuracy

# # Path to the test image
# test_image_path = r"C:\Users\kavya\OneDrive\Desktop\unet 1\test_original\23.jpg"
# test_mask_path = r"C:\Users\kavya\OneDrive\Desktop\unet 1\test_mask\23.jpg" # Ground truth mask path

# # Define image size
# image_size = (64, 64)

# # Preprocess the test image
# test_image = preprocess_test_image(test_image_path, image_size)

# if test_image is not None:
#     # Normalize test image pixel values to range [0, 1]
#     test_image = test_image / 255.0

#     try:
#         # Perform prediction using the model
#         predicted_mask = model.predict(test_image)

#         # Load the ground truth mask
#         ground_truth_mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)
#         ground_truth_mask = cv2.resize(ground_truth_mask, image_size)
#         ground_truth_mask = ground_truth_mask / 255.0  # Normalize to [0, 1]

#         # Calculate test accuracy
#         accuracy = calculate_test_accuracy(predicted_mask.squeeze(), ground_truth_mask)
#         print("Test Accuracy:", accuracy)

#         # Visualize the original test image and the predicted mask
#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 2, 1)
#         plt.imshow(cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB))
#         plt.title("Test Image")
#         plt.axis('off')

#         plt.subplot(1, 2, 2)
#         plt.imshow(predicted_mask.squeeze(), cmap='gray')
#         plt.title("Predicted Mask")
#         plt.axis('off')

#         plt.show()

#     except Exception as e:
#         print(f"Error processing image: {e}")


import cv2                #with grd truth
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model(r"C:\Users\kavya\OneDrive\Desktop\unet 1\unet_model33.h5")

# Function to preprocess a single test image
def preprocess_test_image(image_path, image_size):
    try:
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            return img
        else:
            print("Error: Image not loaded successfully.")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to calculate test accuracy
def calculate_test_accuracy(predicted_mask, ground_truth_mask):
    # Threshold the predicted mask
    predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8)

    # Calculate accuracy
    accuracy = np.mean((predicted_mask_binary == ground_truth_mask).astype(np.float32))
    return accuracy

# Path to the test image
test_image_path = r"C:\Users\kavya\OneDrive\Desktop\unet 1\test_original\99.jpg"
test_mask_path = r"C:\Users\kavya\OneDrive\Desktop\unet 1\test_mask\99.jpg" # Ground truth mask path

# Define image size
image_size = (64, 64)

# Preprocess the test image
test_image = preprocess_test_image(test_image_path, image_size)

if test_image is not None:
    # Normalize test image pixel values to range [0, 1]
    test_image = test_image / 255.0

    try:
        # Perform prediction using the model
        predicted_mask = model.predict(test_image)

        # Load the ground truth mask
        ground_truth_mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)
        ground_truth_mask = cv2.resize(ground_truth_mask, image_size)
        ground_truth_mask = ground_truth_mask / 255.0  # Normalize to [0, 1]

        # Calculate test accuracy
        accuracy = calculate_test_accuracy(predicted_mask.squeeze(), ground_truth_mask)
        print("Test Accuracy:", accuracy)

        # Visualize the original test image, ground truth mask, and the predicted mask
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB))
        plt.title("Test Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(ground_truth_mask, cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(predicted_mask.squeeze(), cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

        plt.show()

    except Exception as e:
        print(f"Error processing image: {e}")

# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
# import os

# # Load the trained model
# model = load_model(r"C:\Users\kavya\OneDrive\Desktop\unet 1\unet_model33.h5")

# # Function to preprocess a single test image
# def preprocess_test_image(image_path, image_size):
#     try:
#         img = cv2.imread(image_path)
#         if img is not None:
#             img = cv2.resize(img, image_size)
#             img = np.expand_dims(img, axis=0)  # Add batch dimension
#             return img
#         else:
#             print(f"Error: Image not loaded successfully: {image_path}")
#             return None
#     except Exception as e:
#         print(f"Error: {e}")
#         return None

# # Function to calculate test accuracy
# def calculate_test_accuracy(predicted_mask, ground_truth_mask):
#     # Threshold the predicted mask
#     predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8)

#     # Calculate accuracy
#     accuracy = np.mean((predicted_mask_binary == ground_truth_mask).astype(np.float32))
#     return accuracy

# # Function to process images in a directory and calculate overall accuracy
# def process_directory(image_dir, mask_dir, image_size):
#     image_files = os.listdir(image_dir)
#     accuracies = []

#     for image_file in image_files:
#         image_path = os.path.join(image_dir, image_file)
#         mask_path = os.path.join(mask_dir, image_file)  # Assuming mask has the same file name

#         # Preprocess the test image
#         test_image = preprocess_test_image(image_path, image_size)

#         if test_image is not None:
#             # Normalize test image pixel values to range [0, 1]
#             test_image = test_image / 255.0

#             try:
#                 # Perform prediction using the model
#                 predicted_mask = model.predict(test_image)

#                 # Load the ground truth mask
#                 ground_truth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#                 if ground_truth_mask is not None:
#                     ground_truth_mask = cv2.resize(ground_truth_mask, image_size)
#                     ground_truth_mask = ground_truth_mask / 255.0  # Normalize to [0, 1]

#                     # Calculate test accuracy
#                     accuracy = calculate_test_accuracy(predicted_mask.squeeze(), ground_truth_mask)
#                     accuracies.append(accuracy)

#                     # Visualize the original test image and the predicted mask
#                     plt.figure(figsize=(10, 5))
#                     plt.subplot(1, 2, 1)
#                     plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
#                     plt.title("Test Image")
#                     plt.axis('off')

#                     plt.subplot(1, 2, 2)
#                     plt.imshow(predicted_mask.squeeze(), cmap='gray')
#                     plt.title("Predicted Mask")
#                     plt.axis('off')

#                     plt.show()
#                 else:
#                     print(f"Error: Ground truth mask not loaded successfully for {image_file}")

#             except Exception as e:
#                 print(f"Error processing image {image_file}: {e}")
#         else:
#             print(f"Skipping image {image_file} due to preprocessing error.")

#     if accuracies:
#         overall_accuracy = np.mean(accuracies)
#         print("Overall Test Accuracy:", overall_accuracy)
#     else:
#         print("No valid images to process for accuracy calculation.")

# # Define image size and directories
# image_size = (64, 64)
# test_image_dir = r"C:\Users\kavya\OneDrive\Desktop\unet 1\test_original"
# test_mask_dir = r"C:\Users\kavya\OneDrive\Desktop\unet 1\test_mask"
# # Process the directory and calculate overall accuracy
# process_directory(test_image_dir, test_mask_dir, image_size)
