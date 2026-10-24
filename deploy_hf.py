"""Create the Hugging Face Space, set the GEMINI_API_KEY secret from .env, and upload the project.

Usage (from the project folder):
    set HF_TOKEN=<token with write access>        (PowerShell: $env:HF_TOKEN="<token>")
    python deploy_hf.py <your-hf-username> [space-name]
"""
import os
import sys
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi

if len(sys.argv) < 2:
    sys.exit("Usage: python deploy_hf.py <your-hf-username> [space-name]")
repo = f"{sys.argv[1]}/{sys.argv[2] if len(sys.argv) > 2 else 'waste-agent'}"

token = os.environ.get("HF_TOKEN")
if not token:
    sys.exit("HF_TOKEN is not set. Create a token with write access at huggingface.co/settings/tokens and set it first.")
key = dotenv_values(Path(__file__).parent / ".env").get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not key or key.startswith("your-"):
    sys.exit("No real GEMINI_API_KEY found in .env.")

api = HfApi(token=token)
api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
api.add_space_secret(repo, "GEMINI_API_KEY", key)  # stored as a secret; never shown on the Space page
api.upload_folder(
    folder_path=str(Path(__file__).parent), repo_id=repo, repo_type="space",
    ignore_patterns=[".git/*", ".venv/*", ".env", "data/*", "notebooks/*", "__pycache__/*", "*.log", "deploy_hf.py", "PROJECT_JOURNEY.md"],
)
print(f"\nUploaded. The build starts now (about 10 minutes).\nBuild logs: https://huggingface.co/spaces/{repo}\nLive URL:   https://{repo.replace('/', '-').lower()}.hf.space")
