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
    """Return (label, probs) where probs is a dict of class->probability."""
    x = torch.tensor(np.array(img), dtype=torch.float32)
    x = x.unsqueeze(0).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy().tolist()
        predicted = int(torch.argmax(logits, dim=1).item())
    prob_map = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return CLASS_NAMES[predicted], prob_map

# ---- Flask Web interface ----
app = Flask(__name__)  

@app.route("/", methods=["GET", "POST"])
def upload_and_classify():
    label = None
    probs = None
    error = None
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename.strip() == "":
            error = "Please choose an image file to upload."
        else:
            try:
                img = Image.open(io.BytesIO(file.read())).convert("L").resize((128, 128))
                label, probs = predict_image(img)
            except Exception:
                error = "That file doesn't look like a valid image. Please upload a PNG/JPG." 
    return render_template("form.html", label=label, probs=probs, error=error)

if __name__ == '__main__':
    model = load_model()
    app.run(host="0.0.0.0", port=5000, debug=True)