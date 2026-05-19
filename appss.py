# Run this Python script once from your PC
from huggingface_hub import HfApi

api = HfApi()

# Create repo for BLIP
api.create_repo("shubmrj/gpt2-rocstories-finetuned", private=False)

# Upload all GPT-2 files
api.upload_folder(
    folder_path = "Models/gpt2",
    repo_id     = "shubmrj/gpt2-rocstories-finetuned",
    repo_type   = "model"
)
print("✅ GPT-2 uploaded")