from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
# Load model
model = joblib.load("image_auth_model.pkl")

# Ask user for image
image_path = input("Enter image path: ")

# Read image
img = cv2.imread(image_path)

if img is None:
    print("❌ Image not found!")
    exit()

# Show image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Input Image")
plt.axis("off")
plt.show()

# Prepare for prediction
img_resized = cv2.resize(img, (128, 128))
img_flat = img_resized.flatten().reshape(1, -1)

# Predict
prediction = model.predict(img_flat)
probability = model.predict_proba(img_flat)

confidence = max(probability[0]) * 100

if prediction[0] == 0:
    print("Prediction: AUTHENTIC IMAGE ✅")
else:
    print("Prediction: TAMPERED IMAGE ❌")

print("Confidence Score:", round(confidence,2), "%")
# LIME Explanation
def predict_lime(images):
    images = np.array(images)
    images = images.reshape(len(images), -1)
    return model.predict_proba(images)

explainer = lime_image.LimeImageExplainer()

explanation = explainer.explain_instance(
    img_resized,
    predict_lime,
    top_labels=1,
    hide_color=0,
    num_samples=100
)

temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=5,
    hide_rest=False
)

plt.imshow(mark_boundaries(temp/255.0, mask))
plt.title("LIME Explanation")
plt.axis("off")
plt.show()