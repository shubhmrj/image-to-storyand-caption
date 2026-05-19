import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from flask import Flask, request, jsonify, render_template
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    GPT2LMHeadModel, GPT2Tokenizer
)
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

BLIP_PATH = "Models/blip"
GPT2_PATH = "Models/gpt2"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load BLIP ─────────────────────────────────────────────────
logger.info("Loading BLIP from %s...", BLIP_PATH)
blip_processor = BlipProcessor.from_pretrained(BLIP_PATH)
blip_model     = BlipForConditionalGeneration.from_pretrained(BLIP_PATH).to(DEVICE)
blip_model.eval()
logger.info("BLIP ready on %s", DEVICE)

# ── Load GPT-2 ────────────────────────────────────────────────
logger.info("Loading GPT-2 from %s...", GPT2_PATH)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(GPT2_PATH)
gpt2_model     = GPT2LMHeadModel.from_pretrained(GPT2_PATH).to(DEVICE)
gpt2_model.eval()
logger.info("GPT-2 ready on %s", DEVICE)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_caption(image):
    """Image → Caption using BLIP"""
    inputs = blip_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = blip_model.generate(
            **inputs,
            max_new_tokens     = 50,
            num_beams          = 5,
            early_stopping     = True,
            repetition_penalty = 1.2
        )
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    if caption:
        caption = caption.capitalize()
        if not caption.endswith(('.', '!', '?')):
            caption += '.'
    return caption

def get_story(caption):
    """Caption → Story using GPT-2"""
    input_text = f"<|story|> {caption}."
    input_ids  = gpt2_tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = gpt2_model.generate(
            input_ids,
            max_new_tokens     = 200,
            temperature        = 0.85,
            top_p              = 0.92,
            do_sample          = True,
            repetition_penalty = 1.3,
            pad_token_id       = gpt2_tokenizer.eos_token_id,
            eos_token_id       = gpt2_tokenizer.encode("<|endofstory|>")[0]
        )

    full_text = gpt2_tokenizer.decode(output[0], skip_special_tokens=False)
    if "<|story|>" in full_text:
        full_text = full_text.split("<|story|>")[-1]
    if "<|endofstory|>" in full_text:
        full_text = full_text.split("<|endofstory|>")[0]
    return full_text.strip().capitalize()


@app.route("/")
def home():
    return render_template("index.html")


# ── /caption  →  Single Photo tab (caption only) ─────────────
@app.route("/caption", methods=["POST"])
def caption():
    if "image" not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    file = request.files["image"]
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid or missing image file"}), 400
    try:
        image   = Image.open(io.BytesIO(file.read())).convert("RGB")
        cap     = get_caption(image)
        return jsonify({"caption": cap})
    except Exception as e:
        logger.error("Caption error: %s", str(e))
        return jsonify({"error": "Issue processing your image. Please try another."}), 500


# ── /story  →  Story Mode tab (caption + story) ───────────────
@app.route("/story", methods=["POST"])
def story():
    if "image" not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    file = request.files["image"]
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid or missing image file"}), 400
    try:
        image   = Image.open(io.BytesIO(file.read())).convert("RGB")
        cap     = get_caption(image)
        story   = get_story(cap)
        return jsonify({"caption": cap, "story": story})
    except Exception as e:
        logger.error("Story error: %s", str(e))
        return jsonify({"error": "Issue generating story. Please try another image."}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)