import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

data = []
labels = []

# Load authentic images
for file in os.listdir("dataset/authentic"):
    img = cv2.imread(os.path.join("dataset/authentic", file))
    img = cv2.resize(img, (128, 128))
    data.append(img.flatten())
    labels.append(0)

# Load tampered images
for file in os.listdir("dataset/tampered"):
    img = cv2.imread(os.path.join("dataset/tampered", file))
    img = cv2.resize(img, (128, 128))
    data.append(img.flatten())
    labels.append(1)

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Performance")
print("---------------------")
print("Accuracy:", round(accuracy*100,2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

joblib.dump(model, "image_auth_model.pkl")

print("\nModel saved as image_auth_model.pkl")