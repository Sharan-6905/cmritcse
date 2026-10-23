---
title: Waste Agent
emoji: ♻️
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Waste Agent

**Waste Classification and Recycling Optimization Using Agentic AI and Computer Vision**

Upload a photo of waste. The agent finds each item, identifies the material, tells you which bin it goes in and how to dispose of it with the least harm to the environment, and shows how much more you would recover by preparing the items properly (rinsing, separating lids, tearing off greasy parts).

## What it does

- **Computer vision:** a ResNet50 trained on 2,527 photos (cardboard, glass, metal, paper, plastic, trash) classifies each item. Grad-CAM heatmaps show where the network looked.
- **Agentic AI:** a Gemini chat agent detects items, checks the classifier against the photo (and overrides it when it is wrong or the item is outside its six classes), and calls tools for the disposal guide and the optimizer. It covers 11 categories: the six above plus organic, battery, e-waste, textile and hazardous.
- **Recycling optimization:** plain code compares "as-is" with "after prep steps" and reports the recovered share of waste and the CO₂ avoided (estimates from typical item weights).
- **Interface:** a chat page with live progress of the agent's steps, boxes drawn on your photo, per-item cards, a session score, and an Insights page with the results.

## Results

| Measure | Result |
|---|---|
| Held-out test accuracy, Colab run (380 photos) | **93.4%** |
| Same code, local GPU run (same split) | 91.8% |
| ResNet50 alone on 19 web photos (its own classes) | 63% |
| Full agent on the same 19 web photos | 84% |
| Full agent on 29 web photos incl. battery, e-waste, food | 83% (24 of 29, strict scoring) |
| Hazardous items found (5 photos) | 5 of 5 (smoke test only) |

The web photos are openly licensed images from Wikimedia Commons (sources in `data/real_world/SOURCES.csv`). The set is small, so treat the percentages as rough. The classifier falls from 93% to 63% on cluttered photos; the agent's second look with Gemini recovers most of that. See [`PROJECT_JOURNEY.md`](PROJECT_JOURNEY.md) for how the project evolved from the first notebook, including the flaws found in it.

## How it works

```
photo -> Gemini vision (find items + boxes) -> ResNet50 per item (+ Grad-CAM)
      -> Gemini checks and decides final categories
      -> tools: disposal guide, recycling optimizer
      -> short plan + cards in the page
```

| Path | Purpose |
|---|---|
| `src/train.py` | Training script (stratified 70/15/15 split, class-weighted loss) |
| `notebooks/train_colab.ipynb` | Colab notebook with all the plots |
| `src/classifier.py` | Loads the ResNet50 and classifies, optionally with a heatmap |
| `src/agent.py` | Gemini chat agent, tools, model fallback |
| `src/optimizer.py` | As-is vs after-prep recovery and CO₂ |
| `src/knowledge.py` | Disposal guides for the 11 categories |
| `src/api.py` | FastAPI app with streaming progress and rate limiting |
| `static/` | Chat page and Insights page |
| `weights/` | Trained weights, class list, metrics |
| `data/` | Training photos and the real-world test photos |
| `Dockerfile` | One-command container build |

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\pip install torch torchvision
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env      # add your Google AI Studio key as GEMINI_API_KEY
.venv\Scripts\python -m uvicorn src.api:app --port 8000
```

Open <http://127.0.0.1:8000>. To retrain: `python -m src.train --data <folder with one sub-folder per class>`.

Useful settings (environment variables): `GEMINI_MODEL` and `GEMINI_MODELS` (model order; slow or out-of-quota models are skipped automatically), `GEMINI_TIMEOUT_S`, `RATE_LIMIT_PER_HOUR`, `DAILY_REQUEST_CAP`.

## Run with Docker (nothing else to install)

The image installs everything itself (CPU PyTorch and the Python packages). The only thing you supply is a Google AI Studio key:

```bash
docker build -t waste-agent .
docker run -p 7860:7860 -e GEMINI_API_KEY=your-key waste-agent
```

Open <http://localhost:7860>. Without a key the page loads, but chatting returns an error.

## Deploy (Hugging Face Spaces)

1. Create a free account at huggingface.co and a **New Space**: SDK **Docker**, hardware **CPU basic**.
2. In the Space, open **Settings > Variables and secrets** and add a secret named `GEMINI_API_KEY` with your key. Never commit the key.
3. Upload the project to the Space with the `huggingface_hub` library (it handles the 90 MB weights file, no Git LFS needed). Create an access token with **write** permission at huggingface.co/settings/tokens, then:
   ```bash
   pip install huggingface_hub
   set HF_TOKEN=<your-token>        # on macOS/Linux: export HF_TOKEN=<your-token>
   python -c "from huggingface_hub import HfApi; a=HfApi(); a.create_repo('<your-user>/waste-agent', repo_type='space', space_sdk='docker', exist_ok=True); a.upload_folder(folder_path='.', repo_id='<your-user>/waste-agent', repo_type='space', ignore_patterns=['.git/*','.venv/*','.env','data/*','__pycache__/*','*.log'])"
   ```
4. Wait for the build (about 10 minutes). The app is then live at `https://<your-user>-waste-agent.hf.space`.

The header at the top of this README (`sdk: docker`, `app_port: 7860`) is what tells Spaces how to run the container. A visitor is limited to 40 requests an hour and the whole site to 250 a day, so strangers cannot use up the Gemini quota.

## Dataset

`data/raw/` holds the 2,527 training photos, one folder per class (cardboard 403, glass 501, metal 410, paper 594, plastic 482, trash 137). The images and class counts match the public **TrashNet** dataset (Thung and Yang, MIT licence); please credit the original authors. `data/real_world/` holds the openly licensed Wikimedia Commons photos used for the real-world test, with authors and licences in `SOURCES.csv`.

## Limits

- The classifier was trained on about 2.5k clean single-item photos from one dataset, so cluttered photos score lower.
- Item detection and the final check use the Gemini API (the free tier has daily limits). Photos are sent to Google for analysis.
- CO₂ and recovery figures use typical weights and simple contamination rules: estimates, not measurements.
- Hazardous items were checked on five photos, textiles on none. Disposal advice is general; check your local rules.
