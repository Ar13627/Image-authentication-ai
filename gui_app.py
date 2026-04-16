import tkinter as tk
from tkinter import filedialog
import cv2
import joblib
import numpy as np
from PIL import Image, ImageTk, ExifTags
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops

# Load model
model = joblib.load("image_auth_model.pkl")


# ---------- NOISE ----------
def calculate_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def get_noise_percentage(noise_value):
    return min((noise_value / 1000) * 100, 100)


# ---------- LIME PREDICT ----------
def lime_predict(images):
    processed = []

    for img in images:
        img = cv2.resize(img, (128,128))
        img = img.flatten()
        processed.append(img)

    return model.predict_proba(np.array(processed))


# ---------- METADATA CHECK ----------
def check_metadata(path):
    try:
        img = Image.open(path)
        exif = img._getexif()

        if exif is None:
            return "No metadata (possible editing)"

        tags = {}
        for tag, value in exif.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            tags[decoded] = value

        if "Software" in tags:
            return f"Edited using: {tags['Software']}"

        return "Metadata looks normal"

    except:
        return "Metadata unavailable"


# ---------- MAIN ----------
def upload_image():

    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = cv2.imread(file_path)

    # ---------- SHOW IMAGE ----------
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_display = cv2.resize(img_display, (250,250))

    img_tk = ImageTk.PhotoImage(Image.fromarray(img_display))
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # ---------- MODEL ----------
    img_resized = cv2.resize(img, (128,128))
    img_flat = img_resized.flatten().reshape(1,-1)

    prediction = model.predict(img_flat)
    probability = model.predict_proba(img_flat)

    confidence = max(probability[0]) * 100

    # ---------- NOISE ----------
    noise_value = calculate_noise(img)
    noise_percent = get_noise_percentage(noise_value)

    # ---------- METADATA ----------
    metadata_result = check_metadata(file_path)

    # ---------- LIME ----------
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (128,128))

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        img_rgb,
        lime_predict,
        top_labels=2,
        hide_color=0,
        num_samples=300
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=5,
        hide_rest=False
    )

    # ---------- FAKE AREA % ----------
    fake_pixels = np.sum(mask != 0)
    total_pixels = mask.size
    fake_percent = (fake_pixels / total_pixels) * 100

    # ---------- IMPORTANT REGION ----------
    regions = regionprops(mask.astype(int))
    if regions:
        largest = max(regions, key=lambda r: r.area)
        minr, minc, maxr, maxc = largest.bbox
        cv2.rectangle(temp, (minc,minr), (maxc,maxr), (255,0,0), 2)

    # ---------- LIME IMAGE ----------
    lime_img = mark_boundaries(temp/255.0, mask)
    lime_img = (lime_img * 255).astype(np.uint8)
    lime_img = cv2.resize(lime_img, (250,250))

    lime_tk = ImageTk.PhotoImage(Image.fromarray(lime_img))
    lime_label.config(image=lime_tk)
    lime_label.image = lime_tk

    # ---------- RESULT LOGIC ----------
    if prediction[0] == 0:
        result = "AUTHENTIC IMAGE"
        explanation_text = "Image appears natural with consistent patterns."

        result_label.config(
            text=f"Prediction: {result}\n"
                 f"Confidence: {confidence:.2f}%\n"
                 f"{metadata_result}"
        )

    else:
        result = "TAMPERED IMAGE"

        explanation_text = (
            f"Suspicious regions detected.\n"
            f"Tampered Area: {fake_percent:.2f}%\n"
            f"Noise Level: {noise_percent:.2f}%"
        )

        result_label.config(
            text=f"Prediction: {result}\n"
                 f"Confidence: {confidence:.2f}%\n"
                 f"Tampered Area: {fake_percent:.2f}%\n"
                 f"Noise: {noise_percent:.2f}%\n"
                 f"{metadata_result}"
        )

    explanation_label.config(text=explanation_text)


# ---------- GUI ----------
root = tk.Tk()
root.title("AI Image Authentication System")
root.geometry("420x720")

title = tk.Label(root, text="AI Image Authentication System", font=("Arial",16))
title.pack(pady=10)

btn = tk.Button(root, text="Upload Image", command=upload_image)
btn.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

lime_label = tk.Label(root)
lime_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial",12))
result_label.pack(pady=10)

explanation_label = tk.Label(
    root,
    text="",
    font=("Arial",11),
    wraplength=350,
    justify="center"
)
explanation_label.pack(pady=10)

root.mainloop()