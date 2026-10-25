"""Loads the trained ResNet50 once and classifies a single image (optionally with a Grad-CAM heatmap)."""
import base64
import io
import json
import os
import threading
from functools import lru_cache
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# On a shared cloud host PyTorch sees every core of the big machine but the container gets only 1-2, so too many
# threads make inference crawl (about 75 s per image on Railway). Cap them there; locally keep PyTorch's default.
_threads = int(os.getenv("TORCH_THREADS") or (2 if any(k.startswith("RAILWAY_") for k in os.environ) else 0))
if _threads:
    torch.set_num_threads(_threads)

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"
_tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


@lru_cache(maxsize=1)
def _load():
    classes = json.loads((WEIGHTS / "classes.json").read_text())
    weights = WEIGHTS / "best_model.pth"
    if weights.stat().st_size < 1024 and weights.read_bytes().startswith(b"version https://git-lfs"):
        raise RuntimeError("weights/best_model.pth is only a Git LFS placeholder. Run: git lfs install && git lfs pull")
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(WEIGHTS / "best_model.pth", map_location="cpu"))
    return model.eval(), classes


_lock = threading.Lock()


def _heatmap(img: Image.Image, cam: torch.Tensor) -> str:
    """Overlay a Grad-CAM map on the 224x224 image and return it as a JPEG data URL."""
    cam = torch.nn.functional.interpolate(cam[None, None], size=(224, 224), mode="bilinear")[0, 0]
    cam = (cam / (cam.max() + 1e-8)).detach().numpy()
    heat = (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    base = np.asarray(img.convert("RGB").resize((224, 224)), dtype=np.float32)
    out = Image.fromarray((0.55 * base + 0.45 * heat).astype(np.uint8))
    buf = io.BytesIO(); out.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def classify(image_path: str, top_k: int = 3, cam: bool = False) -> dict:
    model, classes = _load()
    img = Image.open(image_path).convert("RGB")
    x = _tf(img).unsqueeze(0)
    heat = None
    if cam:
        with _lock:  # one hook at a time on the shared model
            store = {}
            handle = model.layer4[-1].register_forward_hook(lambda m, i, o: store.update(a=o))
            try:
                with torch.enable_grad():
                    out = model(x)
                    probs = torch.softmax(out, dim=1)[0].detach()
                    grads = torch.autograd.grad(out[0, out.argmax(1).item()], store["a"])[0]
                weights = grads.mean(dim=(2, 3), keepdim=True)
                heat = _heatmap(img, torch.relu((weights * store["a"]).sum(1))[0].detach())
            finally:
                handle.remove()
    else:
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
    top = probs.topk(min(top_k, len(classes)))
    ranked = [{"category": classes[i], "confidence": round(p.item(), 3)} for p, i in zip(top.values, top.indices)]
    result = {"category": ranked[0]["category"], "confidence": ranked[0]["confidence"], "top_k": ranked}
    if heat:
        result["cam"] = heat
    return result
