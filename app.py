import os
import io
from PIL import Image
import torch
from model import PneumoniaDiagnosis
from flask import Flask, request, render_template
import numpy as np 

CHECKPOINT_PATH = os.path.join("checkpoints", "best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"] 

def load_model():
    model = PneumoniaDiagnosis()
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model

def predict_image(img: Image.Image):
    x = torch.tensor(np.array(img), dtype = torch.float32)
    x = x.unsqueeze(0).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        output = model(x)
        _, predicted = torch.max(output, dim=1)
    return CLASS_NAMES[predicted]

app = Flask(__name__)  

@app.route("/", methods=["GET", "POST"])
def upload_and_classify():
    label = None
    if request.method == "POST":
        file = request.files["file"]
        img = Image.open(io.BytesIO(file.read())).convert("L").resize((128,128))
        label = predict_image(img)
    return render_template("form.html", label=label)

if __name__ == '__main__':
    model = load_model()
    app.run(host="0.0.0.0", port=5000, debug=True)