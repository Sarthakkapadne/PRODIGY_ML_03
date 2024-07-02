import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Paths
train_dir = 'dataset/train'
test_dir = 'dataset/test'
submission_file = 'dataset/samplesubmission.csv'

# Parameters
img_size = (64, 64)  # Resize images to 64x64

def load_images_and_labels(folder, img_size):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        label = 0 if 'cat' in filename else 1  # Label cats as 0 and dogs as 1
        try:
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img = img.resize(img_size, Image.ANTIALIAS)
            img = np.array(img).flatten()  # Flatten the image
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)

X_train, y_train = load_images_and_labels(train_dir, img_size)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Load and preprocess test images
def load_test_images(folder, img_size):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img = img.resize(img_size, Image.ANTIALIAS)
            img = np.array(img).flatten()  # Flatten the image
            images.append(img)
            filenames.append(filename)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return np.array(images), filenames

X_test, test_filenames = load_test_images(test_dir, img_size)

# Standardize the test features
X_test = scaler.transform(X_test)

# Predict
y_pred = svm.predict(X_test)

# Create the submission DataFrame
submission = pd.DataFrame({'id': test_filenames, 'label': y_pred})
submission['id'] = submission['id'].str.extract('(\d+)').astype(int)  # Extract the numeric id from filenames
submission = submission.sort_values('id')  # Sort by id

# Save to CSV
submission.to_csv('submission.csv', index=False)
