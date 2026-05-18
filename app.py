import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from flask import Flask, request, jsonify, render_template
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

MODEL_PATH = "Models/blip"

def load_model():
    try:
        logger.info("Loading model from %s...", MODEL_PATH)
        processor = BlipProcessor.from_pretrained(MODEL_PATH)
        model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        logger.info("Model ready on %s", device)
        return processor, model, device
    except Exception as e:
        logger.error("Failed to load model: %s", str(e))
        raise

try:
    processor, model, device = load_model()
except Exception:
    pass

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/caption", methods=["POST"])
def caption():
    if "image" not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files["image"]

    if file.filename == '':
        return jsonify({"error": "No selected image"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Process image
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=5,
                early_stopping=True,
                repetition_penalty=1.2
            )

        generated_caption = processor.decode(output[0], skip_special_tokens=True)
        
        if generated_caption:
            generated_caption = generated_caption.capitalize()
            if not generated_caption.endswith(('.', '!', '?')):
                generated_caption += '.'

        return jsonify({"caption": generated_caption})

    except Exception as e:
        logger.error("Captioning error: %s", str(e))
        return jsonify({"error": "I encountered an issue processing your image. Please try another one."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)