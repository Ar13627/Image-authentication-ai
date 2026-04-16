# 🧠 AI Image Authentication System

An advanced AI-based system that detects whether an image is **Authentic** or **Tampered** using Machine Learning and **LIME Explainable AI**.

---

## 🚀 Features

* 🔍 Image Classification (Authentic / Tampered)
* 📊 Confidence Score Prediction
* 🧠 LIME Explainable AI Visualization
* 🎯 Tampering Region Highlight (Bounding Box)
* 🌫️ Noise Percentage Detection (Only for Tampered Images)
* 🧾 Metadata Analysis (Editing Detection)
* 🖥️ Interactive GUI (Tkinter)

---

## 🧪 Technologies Used

* Python 🐍
* OpenCV
* Scikit-learn
* NumPy
* LIME (Explainable AI)
* Tkinter (GUI)
* PIL (Image Processing)

---

## ⚙️ How It Works

1. User uploads an image
2. Image is resized and converted into feature vector
3. Machine Learning model predicts:

   * Authentic or Tampered
   * Confidence score
4. LIME generates explanation heatmap
5. System:

   * Highlights important regions
   * Detects noise level
   * Checks metadata

---

## 📸 Output Example

* Original Image
* LIME Heatmap (Highlighted Regions)
* Prediction + Confidence
* Noise Percentage (only if tampered)
* Explanation Text

---

## 🧠 Core Concepts

* Image Processing
* Machine Learning Classification
* Feature Extraction (Flattening)
* Explainable AI (LIME)
* Noise Detection using Laplacian Variance
* Metadata Analysis

---

## ▶️ Run the Project

```bash
python gui_app.py
```

---

## 📂 Project Structure

```
ImageAuth/
│
├── dataset/                # Training data
├── train_model.py          # Model training
├── predict.py              # Prediction logic
├── gui_app.py              # Main GUI application
├── explain.py              # LIME explanation
├── image_auth_model.pkl    # Trained model
```

---

## 💡 Unique Features (What Makes This Project Different)

✔ Explainable AI (not just prediction)
✔ Noise % detection for tampering analysis
✔ Visual bounding box of manipulated regions
✔ Metadata-based editing detection
✔ GUI-based real-time system

---

## 📊 Future Improvements

* 🔥 Deep Learning (CNN) integration
* 🌐 Web App version (Flask / Streamlit)
* 📱 Mobile App support
* 🧠 Better tampering localization

---

## 👨‍💻 Author

**Aryan Dhiman**
B.Tech CSE Student
Passionate about AI, Security & Software Development
---
assets/original.png
assets/lime_output.png
