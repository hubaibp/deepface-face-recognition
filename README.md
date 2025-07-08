# 👨‍🏫 Student Face Recognition Attendance System

A real-time face recognition attendance system built using **Django**, **OpenCV**, and **DeepFace**.  
It captures student faces via webcam, encodes them using deep learning, and stores them in a `.pkl` file for persistent and scalable recognition.

---

## 🧠 How It Works

- 🧍 **Capture Faces**  
  Open webcam and capture student images.

- ✍️ **Register Name**  
  Enter student details (name, role, department).

- 📦 **Save Encoding**  
  The system automatically creates `face_data.pkl` using **Joblib**, which stores facial embeddings and associated names.

- 👁️ **Recognize Faces**  
  Match real-time webcam input against the saved `.pkl` face database.

---

## 💾 What is `face_data.pkl`?

- Auto-generated using **Joblib**
- Stores **face embeddings** (numerical vectors)
- Links each embedding to a **student name**
- Acts as a lightweight **face database**
- Enables **persistent**, **scalable** recognition even after restarts

---

## 🧰 Technologies & Their Purpose

| 🛠️ Technology         | 💡 Purpose                                                  |
|------------------------|-------------------------------------------------------------|
| **OpenCV**             | Webcam access & face detection via Haar Cascade             |
| **DeepFace (Facenet)** | Face encoding using a pre-trained deep learning model       |
| **Scikit-learn**       | Normalize face embeddings for accurate comparison           |
| **NumPy**              | Compute distances between embeddings                        |
| **Joblib**             | Save/load face encodings persistently using `face_data.pkl` |

---

## 🚀 Features

- 🎥 Real-time face detection using webcam  
- 💡 Deep learning-based encoding with **Facenet (DeepFace)**  
- 💾 Persistent storage with `.pkl` file  
- 📊 View attendance records by date and student   
- ⚡ High speed and accuracy in recognition



## ⚙️ How to Run Locally

```bash
# Clone the repository

cd student-attendance-face

# Create a virtual environment
python -m venv venv

# Activate the environment (Windows)
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Or manually:
pip install django opencv-python deepface numpy scikit-learn joblib

# Run the development server
python manage.py runserver
