import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    GPT2LMHeadModel, GPT2Tokenizer, TextIteratorStreamer
)
from PIL import Image
import io
from threading import Thread
from deep_translator import GoogleTranslator


app = Flask(__name__, template_folder='Templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
GPT2_MODEL_ID = "gpt2"

# ── Load BLIP ─────────────────────────────────────────────────
logger.info("Loading BLIP from %s...", BLIP_MODEL_ID)
blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
blip_model     = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID)
blip_model.eval()
logger.info("BLIP ready")

# ── Load GPT-2 ────────────────────────────────────────────────
logger.info("Loading GPT-2 from %s...", GPT2_MODEL_ID)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_ID)
if gpt2_tokenizer.pad_token is None:
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

gpt2_model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_ID)
gpt2_model.eval()
logger.info("GPT-2 ready")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_caption(image):
    """Image → Caption using BLIP"""
    inputs = blip_processor(images=image, return_tensors="pt")
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

def clean_story(full_text, caption):
    """Clean the generated story text"""
    for token in ["<|story|>", "<|endofstory|>", "<|pad|>", "<|endoftext|>"]:
        full_text = full_text.replace(token, "")

    story = full_text.strip()
    
    # Remove prompt
    prompt = f"A creative and engaging story about: {caption}"
    if prompt.lower() in story.lower():
        idx = story.lower().find(prompt.lower())
        story = story[:idx] + story[idx + len(prompt):]

    story = story.strip()
    while story and story[0] in ".,!?;: ":
        story = story[1:].strip()
        
    if not story:
        return "The image inspired a quiet, thoughtful moment that words couldn't quite capture yet."
        
    return story.capitalize()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/caption", methods=["POST"])
def caption():
    if "image" not in request.files:
        return jsonify({"error": "No image part"}), 400
    file = request.files["image"]
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid image"}), 400
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        cap = get_caption(image)
        return jsonify({"caption": cap})
    except Exception as e:
        logger.error("Caption error: %s", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/story-stream")
def story_stream():
    caption = request.args.get('caption', '')
    continue_text = request.args.get('continue_text', '')
    
    if continue_text:
        input_text = f"{continue_text} Then,"
    else:
        input_text = f"A creative and engaging story about: {caption}"

    input_ids = gpt2_tokenizer.encode(input_text, return_tensors="pt")
    streamer = TextIteratorStreamer(gpt2_tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=150,
        temperature=0.85,
        top_p=0.92,
        do_sample=True,
        repetition_penalty=1.3,
        pad_token_id=gpt2_tokenizer.pad_token_id,
        eos_token_id=gpt2_tokenizer.eos_token_id
    )

    thread = Thread(target=gpt2_model.generate, kwargs=generation_kwargs)
    thread.start()

    def generate():
        for new_text in streamer:
            yield f"data: {json.dumps({'text': new_text})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    text = data.get('text', '')
    target_lang = data.get('lang', 'en')
    
    if not text:
        return jsonify({"error": "No text"}), 400
        
    try:
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return jsonify({"translated": translated})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
