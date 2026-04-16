import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Load trained model
model = joblib.load("image_auth_model.pkl")

def predict_fn(images):
    images = np.array([cv2.resize(img, (128,128)).flatten() for img in images])
    return model.predict_proba(images)

# Input image
image_path = input("Enter image path for explanation: ")
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create LIME explainer
explainer = lime_image.LimeImageExplainer()

explanation = explainer.explain_instance(
    image,
    predict_fn,
    top_labels=1,
    hide_color=0,
    num_samples=1000
)

temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=5,
    hide_rest=False
)

plt.imshow(mark_boundaries(temp / 255.0, mask))
plt.title("LIME Explanation")
plt.axis("off")
plt.show()