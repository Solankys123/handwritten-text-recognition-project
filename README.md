<h1 align="center">🧠 Handwritten Text Recognition (CRNN)</h1>

<p align="center">
  <b>A deep learning web app that recognizes handwritten English text using a CRNN model trained on the IAM Handwriting Dataset.</b><br>
  Built with ❤️ using <a href="https://pytorch.org/" target="_blank">PyTorch</a> and <a href="https://streamlit.io/" target="_blank">Streamlit</a>.
</p>

---

<p align="center">
  <a href="https://handwritten-text-recognition-project-lb2crqdyjphfksgpj8mswc.streamlit.app/"><img src="https://img.shields.io/badge/🌐_Live_Demo-Click_Here-success?style=for-the-badge&logo=streamlit"></a>
  <a href="https://github.com/Solankys123/handwritten-text-recognition-project/stargazers"><img src="https://img.shields.io/github/stars/Solankys123/handwritten-text-recognition-project?style=for-the-badge&color=yellow"></a>
  <a href="https://github.com/Solankys123/handwritten-text-recognition-project/issues"><img src="https://img.shields.io/github/issues/Solankys123/handwritten-text-recognition-project?style=for-the-badge&color=red"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"></a>
</p>

---

## ✨ Overview

This project implements a **Convolutional Recurrent Neural Network (CRNN)** for recognizing handwritten text from images.  
It combines **CNNs** (for feature extraction) and **BiLSTMs** (for sequence modeling) with a **CTC loss layer** for transcription.  

The model was trained on the **IAM Handwriting Dataset** containing over **38,000 labeled handwritten samples**.  
A **Streamlit-based web interface** allows users to upload any handwritten image and instantly view recognized text.

---

## 🚀 Live Demo
👉 **[Click here to try it out!](https://handwritten-text-recognition-project-lb2crqdyjphfksgpj8mswc.streamlit.app/)**  

*(Upload any handwritten image and see the recognized text in real time!)*

---

## 🧩 Features
✅ Trained on **IAM Handwriting Dataset (~38k samples)**  
✅ Real-time text recognition with **Streamlit UI**  
✅ Auto-downloads pre-trained model from Google Drive  
✅ Fast, accurate, and lightweight (FP16 model ~31MB)  
✅ Built entirely with **PyTorch + Streamlit**

---

## 🧠 Model Details

| Component | Description |
|------------|-------------|
| **Architecture** | CRNN (CNN + BiLSTM + CTC) |
| **Dataset** | IAM Handwriting Dataset |
| **Accuracy** | ~97% Validation Accuracy |
| **Frameworks** | PyTorch, Streamlit |
| **Model File** | `crnn_iam_advanced_fp16.pth` |
| **File Size** | ~31 MB |

---

## 🧰 Tech Stack

| Category | Tools |
|-----------|-------|
| 💻 Deep Learning | PyTorch |
| 🎨 Interface | Streamlit |
| 📊 Data Handling | Pandas, Torch Datasets |
| ☁️ Hosting | Streamlit Cloud |
| 🧩 Augmentation | Torchvision Transforms |

---

## ⚡ Run Locally

```bash
# 1️⃣ Clone this repository
git clone https://github.com/Solankys123/handwritten-text-recognition-project.git

# 2️⃣ Go to project folder
cd handwritten-text-recognition-project

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run Streamlit App
streamlit run app.py
