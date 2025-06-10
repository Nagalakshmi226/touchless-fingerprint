# 🧠 Touchless Fingerprint Recognition

A touchless fingerprint verification system that uses a custom CNN model for biometric feature extraction and a GUI interface for user-friendly verification.

## 📌 Features
- 🧠 Deep CNN model to extract fingerprint embeddings
- 📊 ROC AUC-based performance evaluation
- 🖼️ GUI for uploading and checking fingerprint matches
- 💾 Persistent storage of trained models and embeddings

## 📁 Project Structure
- `model.py` — Trains the CNN model and generates embeddings
- `accuracy.py` — Computes ROC AUC and accuracy
- `gui.py` — GUI for user input and verification (Tkinter + PIL + OpenCV)

## 🛠 Requirements

```bash
pip install tensorflow opencv-python scikit-learn
```
✅ This includes:

tensorflow – for CNN model training

opencv-python – for image reading and preprocessing

scikit-learn – for normalization and evaluation metrics
## 🚀 How to Run

```bash
python model.py
python accuracy.py
python gui.py
```

> Make sure the model (`fingerprint_model.keras`) and embeddings (`fingerprint_embeds.pkl`) are in the same directory as the GUI script.
