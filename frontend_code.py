import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow import keras

# Load the trained model
model = keras.models.load_model(r"C:\Users\kavya\OneDrive\Desktop\dataset\pic.csv\model2.keras")

# Define class names
class_names = ["Healthy", "Alternaria", "Bacterial Blight", "Cercospora", "Fruit Rot"]

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_disease(image_path):
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    accuracy = np.max(predictions) * 100  # Get percentage accuracy
    return predicted_class, accuracy

def display_image(file_path):
    img = Image.open(file_path)
    img = img.resize((300, 300))  # Resize image
    img = ImageTk.PhotoImage(img)
    img_label.config(image=img)
    img_label.image = img  # To prevent image from being garbage collected
    img_label.pack()

def display_prediction(predicted_class, accuracy):
    prediction_label.config(text=f"Predicted Disease: {predicted_class}")
    accuracy_label.config(text=f"Accuracy: {accuracy:.2f}%")

def browse_image():
    file_paths = filedialog.askopenfilenames()
    if file_paths:
        global selected_image_paths
        selected_image_paths = list(file_paths)
        predict_button.config(state="normal")  # Enable predict button

def predict_images():
    for image_path in selected_image_paths:
        try:
            predicted_class, accuracy = predict_disease(image_path)
            display_image(image_path)
            display_prediction(predicted_class, accuracy)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create Tkinter window
root = tk.Tk()
root.title("Pomegranate Disease Detection")

# Set background image
background_image = Image.open(r"C:\Users\kavya\OneDrive\Pictures\background_image.jpg")
background_photo = ImageTk.PhotoImage(background_image)
background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Add Heading
heading_label = tk.Label(root, text="Pomegranate Disease Detection", font=("Helvetica", 35))
heading_label.pack(pady=20)

# Browse button
browse_button = tk.Button(root, text="Browse Images", font=("Helvetica", 15), command=browse_image)
browse_button.pack(pady=10)

# Predict button
predict_button = tk.Button(root, text="Predict Diseases", font=("Helvetica", 15), command=predict_images, state="disabled")
predict_button.pack(pady=10)

# Image label
img_label = tk.Label(root)
img_label.pack()

# Prediction labels
prediction_label = tk.Label(root, text="Predicted Disease:", font=("Helvetica", 15))
prediction_label.pack()

accuracy_label = tk.Label(root, text="Accuracy:", font=("Helvetica", 15))
accuracy_label.pack()

root.mainloop()


