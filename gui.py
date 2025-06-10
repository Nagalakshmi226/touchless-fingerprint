import tkinter as gui
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as math
from keras.models import load_model
from sklearn.preprocessing import normalize
import cv2 as vision
import pickle

verification_model = load_model("fingerprint_model.keras")
with open("fingerprint_embeds.pkl", "rb") as f:
    embed_data = pickle.load(f)

stored_embeds = normalize(embed_data['vectors'])
stored_labels = embed_data['labels']

def process_image():
    file = filedialog.askopenfilename(title="Select Fingerprint")
    if not file:
        return

    img = vision.imread(file, vision.IMREAD_GRAYSCALE)
    if img is None:
        result.config(text="Error: Invalid file", fg="red")
        return

    img_proc = vision.resize(img, (100, 100))
    img_proc = img_proc.reshape(1, 100, 100, 1) / 255.0
    embed = verification_model.predict(img_proc)
    embed_norm = normalize(embed)
    similarity = embed_norm @ stored_embeds.T
    max_score = math.max(similarity)

    result_text = "Match ✅" if max_score > 0.985 else "No Match ❌"
    color = "green" if max_score > 0.985 else "red"

    pil_img = Image.open(file).resize((180, 180))
    bordered_img = Image.new("RGB", (200, 200), color)
    bordered_img.paste(pil_img, (10, 10))

    tk_img = ImageTk.PhotoImage(bordered_img)
    image_label.config(image=tk_img)
    image_label.image = tk_img
    result.config(text=f"{result_text} (Score: {max_score:.4f})", fg=color)

app = gui.Tk()
app.title("Fingerprint Validator")
app.geometry("400x450")
app.configure(bg="#f0f8ff")

header = gui.Label(app, text="Fingerprint Check", font=("Arial", 18, "bold"), bg="#f0f8ff")
header.pack(pady=15)

upload_btn = gui.Button(app, text="Upload Fingerprint", command=process_image, bg="#4B0082", fg="white")
upload_btn.pack(pady=10)

image_label = gui.Label(app, bg="#f0f8ff")
image_label.pack(pady=12)

result = gui.Label(app, text="", font=("Arial", 12), bg="#f0f8ff")
result.pack()

app.mainloop()
