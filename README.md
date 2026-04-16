# 🧠 AI Image Authentication System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![AI](https://img.shields.io/badge/AI-Machine%20Learning-green)
![Explainable AI](https://img.shields.io/badge/LIME-Explainable%20AI-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

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

## 📸 Screenshots

### 🖼️ Original Image

<img width="1913" height="1079" alt="image" src="https://github.com/user-attachments/assets/0c2c991c-d2b1-42a1-9144-4d69b98ff161" />

<img width="497" height="334" alt="image" src="https://github.com/user-attachments/assets/130bed90-2ecc-4bf1-b906-727b2f3f4424" />

### 🧠 LIME Explanation (Tampered Regions Highlighted)

<img width="412" height="585" alt="image" src="https://github.com/user-attachments/assets/1cc9af98-6122-4519-9cd1-e99c6c9ffa0b" />

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
   * Detects noise percentage
   * Checks metadata

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
├── dataset/                
├── train_model.py          
├── predict.py              
├── gui_app.py              
├── explain.py              
├── image_auth_model.pkl    
```

---

## 💡 What Makes This Project Unique

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

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
