from PIL import Image
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def load_and_preprocess_images(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpeg"):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path).convert("L")  # Convert to grayscale
            img = img.resize((64, 64))  # Resize to a consistent size
            img_array = np.array(img).flatten()  # Flatten image into a 1D array
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
russia_images, russia_labels = load_and_preprocess_images(russia_folder, label="Russia")
ukraine_images, ukraine_labels = load_and_preprocess_images(ukraine_folder, label="Ukraine")

# Combine the data from both countries
all_images = russia_images + ukraine_images
all_labels = russia_labels + ukraine_labels

# Convert data to numpy arrays
X = np.array(all_images)
y = np.array(all_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train: Features for training
# y_train: Labels for training
# X_test: Features for testing
# y_test: Labels for testing

# Step 3: Perform PCA.
# Calculate the minimum of n_samples and n_features
min_components = min(X_train.shape[0], X_train.shape[1])

# Choose a reasonable value for n_components (e.g., 150 or less)
n_components = min(min_components, 150)

# Perform PCA with the chosen n_components
pca = PCA(n_components=n_components).fit(X_train)
# Transform training and testing data
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
# Step 5: Initialize Classifier and fit training data
clf = SVC(kernel='rbf', C=1000, gamma=0.001)
clf.fit(X_train_pca, y_train)

# Step 6: Perform testing and get classification report
y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred))

# Ukraine
# Load and preprocess the new image
new_image = load_and_preprocess_image("./data/ukraine/volodemir.jpg")
new_image_pca = pca.transform([new_image])  # Wrap new_image in a list to create a 2D array

# Predict nationality of the new image
predicted_nationality = clf.predict(new_image_pca)
print("Predicted Nationality:", predicted_nationality[0])

# Rusia
# Load and preprocess the new image
new_image = load_and_preprocess_image("./data/russia/vladimir.jpg")
new_image_pca = pca.transform([new_image])  # Wrap new_image in a list to create a 2D array

# Predict nationality of the new image
predicted_nationality = clf.predict(new_image_pca)
print("Predicted Nationality:", predicted_nationality[0])
