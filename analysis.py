from PIL import Image
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics.pairwise import pairwise_distances

def load_and_preprocess_images(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpeg"):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path).convert("L")  # Convert to grayscale
            img = img.resize((64, 64))  # Resize to a consistent size
            # Flatten image into a 1D array
            img_array = np.array(img).flatten()
            images.append(img_array)
            labels.append(label)
    return images, labels


def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize((64, 64))  # Resize to a consistent size
    img_array = np.array(img).flatten()  # Flatten image into a 1D array
    return img_array


# Specify paths to your image folders
russia_folder = "./data/russia/"
ukraine_folder = "./data/ukraine/"

# Load and preprocess images from each folder
russia_images, russia_labels = load_and_preprocess_images(
    russia_folder, label="Russia")
ukraine_images, ukraine_labels = load_and_preprocess_images(
    ukraine_folder, label="Ukraine")

# Combine the data from both countries
all_images = russia_images + ukraine_images
all_labels = russia_labels + ukraine_labels

# Convert data to numpy arrays
X = np.array(all_images)
y = np.array(all_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# X_train: Features for training
# y_train: Labels for training
# X_test: Features for testing
# y_test: Labels for testing

# Train separate PCA models for Russia and Ukraine (same as before)
# Calculate the minimum of n_samples and n_features
n_components = 4  # Adjust this value based on your preference and dataset size

# Train separate PCA models for Russia and Ukraine
pca_russia = PCA(n_components=n_components).fit(russia_images)
pca_ukraine = PCA(n_components=n_components).fit(ukraine_images)

# Transform testing data using PCA models
X_test_russia_pca = pca_russia.transform(X_test[y_test == 'Russia'])
X_test_ukraine_pca = pca_ukraine.transform(X_test[y_test == 'Ukraine'])


def predict_nationality(pca_russia, pca_ukraine, new_image):
    # Transform new image using PCA models
    new_image_russia_pca = pca_russia.transform([new_image])
    new_image_ukraine_pca = pca_ukraine.transform([new_image])

    # Calculate distances between new image and transformed images of each country
    distance_russia = pairwise_distances(new_image_russia_pca, X_test_russia_pca, metric='cosine')
    distance_ukraine = pairwise_distances(new_image_ukraine_pca, X_test_ukraine_pca, metric='cosine')

    # Calculate the minimum distance for each country
    min_distance_russia = np.min(distance_russia)
    min_distance_ukraine = np.min(distance_ukraine)

    # Compare minimum distances and make prediction
    if min_distance_russia < min_distance_ukraine:
        return "Russia"
    else:
        return "Ukraine"

# Load and preprocess the new images
new_image_russia = load_and_preprocess_image("./data/russia/vladimir.jpg")
new_image_ukraine = load_and_preprocess_image("./data/ukraine/volodemir.jpg")

# Predict nationality for each image
predicted_nationality_russia = predict_nationality(pca_russia, pca_ukraine, new_image_russia)
predicted_nationality_ukraine = predict_nationality(pca_russia, pca_ukraine, new_image_ukraine)

print("Predicted Nationality for Vladimir:", predicted_nationality_russia)
print("Predicted Nationality for Volodemir:", predicted_nationality_ukraine)