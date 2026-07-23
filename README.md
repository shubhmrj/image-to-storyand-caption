---
title: Image To Story
emoji: 🖼️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Vision & Words — Image To Story

This application uses BLIP for image captioning and GPT-2 for creative story generation.

## Local Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `python app.py`

## Deployment
Deployed on Hugging Face Spaces using Docker.

# Vision & Words

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/) [![Flask](https://img.shields.io/badge/Flask-3.0.0-000000?logo=flask)](https://flask.palletsprojects.com/) [![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-FFD21F?logo=huggingface)](https://huggingface.co/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C?logo=pytorch)](https://pytorch.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![GitHub Stars](https://img.shields.io/github/stars/shubhmrj/image-to-storyand-caption?style=social)](https://github.com/shubhmrj/image-to-storyand-caption/stargazers) [![GitHub Forks](https://img.shields.io/github/forks/shubhmrj/image-to-storyand-caption?style=social)](https://github.com/shubhmrj/image-to-storyand-caption/network/members)

A lightweight Flask web application that turns an uploaded image into a descriptive caption and a creative story, with optional streaming, translation, and export features.

## 🌐 Overview

Vision & Words is an AI-powered image companion designed to help users describe a picture in natural language and transform that description into an engaging short story. The app uses a vision-language model for caption generation and a language model for storytelling, making it suitable for creative writing, accessibility tools, and interactive demos.

This repository is a practical example of combining computer vision and generative AI in a simple web app that can run locally or in a containerized environment.

## ✨ Features

- Image upload support for JPG, PNG, and WEBP files
- Automatic image caption generation using BLIP
- Creative story generation using GPT-2
- Real-time streaming responses with Server-Sent Events
- Text translation into multiple languages
- Markdown and PDF export for generated content
- Speech synthesis support for accessibility
- Clipboard copy support for quick sharing
- Docker support for containerized deployment
- Structured logging and error handling

## 🏗️ Architecture

The application follows a simple end-to-end workflow:

```text
User
  │
  ▼
Upload Image
  │
  ▼
Image Preprocessing
  │
  ▼
BLIP Caption Model
  │
  ▼
Caption Text
  │
  ▼
GPT-2 Story Generation
  │
  ▼
Streaming UI Output
  │
  ▼
Translation / Export / Speech
```

## 🛠️ Tech Stack

| Category | Technology |
| --- | --- |
| Language | Python 3.10 |
| Backend | Flask |
| ML / LLM | Hugging Face Transformers, BLIP, GPT-2 |
| Deep Learning | PyTorch |
| Image Processing | Pillow |
| Translation | deep-translator |
| Frontend | Vanilla JavaScript, HTML, CSS |
| Deployment | Docker, Gunicorn |
| Libraries | NumPy, Werkzeug |

## 📁 Folder Structure

```text
Captioning/
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
├── docs/
│   └── architecture.md
├── Model/
├── Models/
│   ├── blip/
│   └── gpt2/
├── Notebook/
│   ├── image.ipynb
│   └── main.ipynb
└── Templates/
    └── index.html
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/shubhmrj/image-to-storyand-caption.git
cd image-to-storyand-caption
```

### 2. Create and activate a virtual environment

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

The app will be available at:

```text
http://localhost:7860
```

### 5. Run with Docker

```bash
docker build -t vision-words .
docker run -p 7860:7860 vision-words
```

## 🌍 Environment Variables

This project does not require any mandatory environment variables at runtime. The app uses a small set of defaults and the Hugging Face model cache locally.

If needed, you can optionally set:

```env
HF_HOME=./.cache/huggingface
KMP_DUPLICATE_LIB_OK=TRUE
```

## ▶️ Usage

1. Open the app in your browser.
2. Upload an image in JPG, PNG, or WEBP format.
3. Click the action button to generate a caption.
4. Switch to story mode to generate a creative narrative.
5. Use the streaming output, translation selector, or export buttons as needed.

## 🔌 API Endpoints

| Method | Route | Description |
| --- | --- | --- |
| GET | `/` | Serves the main web interface |
| POST | `/caption` | Accepts an uploaded image and returns a caption |
| GET | `/story-stream` | Streams generated story content using SSE |
| POST | `/translate` | Translates text into a target language |

### Example: Caption generation

```bash
curl -X POST http://localhost:7860/caption \
  -F "image=@/path/to/image.jpg"
```

Example response:

```json
{
  "caption": "A quiet street lined with trees and warm golden light."
}
```

### Example: Story streaming

```bash
curl "http://localhost:7860/story-stream?caption=A%20quiet%20street%20lined%20with%20trees"
```

### Example: Translation

```bash
curl -X POST http://localhost:7860/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"A quiet street lined with trees","lang":"fr"}'
```

## 📸 Screenshots

Screenshots will be added here as the project evolves. Recommended visuals include:

- Upload interface
- Generated caption output
- Story streaming view
- Translation and export controls

## 💬 Example Conversation

User: "Can you describe this photo?"

Assistant: "A quiet street lined with trees and warm golden light."

User: "Tell me a story about it."

Assistant: "The evening breeze carried the scent of rain as the old street shimmered beneath the fading sun."

## 🧠 How the Current Pipeline Works

This repository does not currently implement a PDF-to-answer RAG pipeline or a vector database. Instead, it uses a direct vision-to-language workflow:

1. The uploaded image is opened and converted to RGB.
2. BLIP generates a descriptive caption from the image.
3. The caption is used as a prompt for GPT-2.
4. GPT-2 produces a creative story token by token.
5. The UI streams the result in real time and optionally translates or exports it.

## 🔄 Project Workflow

- Image upload and validation
- Caption generation via BLIP
- Story generation via GPT-2
- Streaming to the frontend
- Optional translation and export
- Logging and graceful error reporting

## ⚡ Performance

The implementation includes several practical performance choices:

- Models are loaded once at startup rather than on every request
- Inference runs with `torch.no_grad()` to reduce memory overhead
- Story streaming uses a background thread and `TextIteratorStreamer` for responsive output
- Beam search and repetition penalties are tuned to improve generation quality

## 🛡️ Error Handling

The application validates file inputs before processing and returns clear errors for invalid uploads or runtime exceptions. Logging is enabled for captioning, translation, and general request failures so issues can be diagnosed quickly.

## 🔮 Future Improvements

Potential enhancements for future versions include:

- Support for multiple image uploads and batch storytelling
- A more advanced prompt engine for richer narratives
- Integration with a modern LLM provider such as OpenAI, Anthropic, or Gemini
- Persistent chat history and user sessions
- Better UI/UX and responsive design
- Optional deployment to cloud platforms with managed GPUs

## 🤝 Contributing

Contributions are welcome. If you would like to improve the project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Open a pull request with a clear description

Please keep changes focused, document new behavior, and ensure the app still runs locally.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgements

This project makes use of:

- Flask
- PyTorch
- Hugging Face Transformers
- Pillow
- deep-translator
- Gunicorn
- Docker
