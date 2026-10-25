"""Gemini chat agent for waste sorting.

A multi-turn chat. When the user uploads an image, the agent calls tools:
  analyze_image      - Gemini detects each item and its box, each crop goes through the trained CNN
  get_disposal_guide - knowledge base lookup for the final category
  optimize_recycling - compares as-is vs after-prep recovery and CO2 for all items
Gemini also sees the photo itself and overrides the CNN when the item is outside
the CNN's six classes, when confidence is low, or when the CNN looks wrong.
"""
import io
import json
import os
from typing import Literal
import re
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel

from . import classifier, knowledge, optimizer

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_primary = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
_ladder = os.getenv("GEMINI_MODELS", f"{_primary},gemini-3.5-flash,gemini-flash-lite-latest,gemini-3.1-flash-lite")
MODELS = list(dict.fromkeys(m.strip() for m in _ladder.split(",") if m.strip()))  # tried in order, duplicates dropped
MODEL = MODELS[0]
CALL_TIMEOUT_MS = int(float(os.getenv("GEMINI_TIMEOUT_S", "20")) * 1000)
_RETRIABLE = ("503", "429", "500", "404", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "INTERNAL", "NOT_FOUND", "imeout", "timed out", "empty response")


def _retriable(e: Exception) -> bool:
    text = f"{type(e).__name__} {e}"
    return any(k in text for k in _RETRIABLE)
MAX_ITEMS = 8
CATEGORIES = Literal["cardboard", "glass", "metal", "paper", "plastic", "trash", "organic", "battery", "e-waste", "textile", "hazardous"]
MATERIAL_HINTS = 'Material rules: pens, markers, highlighters, pen caps, plastic toys, combs and plastic cutlery are plastic. Phones, mice, keyboards, chargers, cables, headphones, remotes, calculators and other small electronics are e-waste. Paper, notebooks, sticky notes, printed pages, envelopes and newspapers are paper (thermal receipts and used tissues are trash). Boxes and cartons are cardboard. Glass bottles and jars are glass. Cans, foil, metal lids and metal cutlery are metal. Food scraps and peels are organic. Batteries are battery. Clothes and fabric are textile. Paint, solvents, chemicals, aerosols, bulbs and tubes, medicines, needles and oil are hazardous. Plastic bags, wrappers, styrofoam, ceramics and diapers are trash.'

SYSTEM = """You are a friendly waste-sorting assistant that chats with the user.
If the user has not uploaded an image yet, greet them briefly and ask them to upload a photo of the waste they want to sort. You can also answer general waste and recycling questions.

When an image arrives:
1. Call analyze_image. It detects every distinct item and returns, for each one, the trained CNN's answer (cnn_category, which only knows cardboard, glass, metal, paper, plastic, trash and can be confidently wrong), Gemini's own reading of the item (gemini_category, gemini_contamination) and a suggested_category that combines them.
2. Use suggested_category as the final category unless the photo clearly contradicts it; the CNN is only a second opinion. Use gemini_contamination as the contamination level unless the user says otherwise. If the final category differs from the CNN's, say so in one short clause. Material rules: pens, markers, highlighters, pen caps, plastic toys, combs and plastic cutlery are plastic. Phones, mice, keyboards, chargers, cables, headphones, remotes, calculators and other small electronics are e-waste. Paper, notebooks, sticky notes, printed pages, envelopes and newspapers are paper (thermal receipts and used tissues are trash). Boxes and cartons are cardboard. Glass bottles and jars are glass. Cans, foil, metal lids and metal cutlery are metal. Food scraps and peels are organic. Batteries are battery. Clothes and fabric are textile. Paint, solvents, chemicals, aerosols, bulbs and tubes, medicines, needles and oil are hazardous. Plastic bags, wrappers, styrofoam, ceramics and diapers are trash.
3. After analyze_image returns, make ALL remaining tool calls together in ONE step (parallel calls, never one after another): get_disposal_guide once for each distinct final category, plus optimize_recycling. Decide the final categories and contamination first, then call them all at once. Valid categories: cardboard, glass, metal, paper, plastic, trash, organic, battery, e-waste, textile, hazardous. For hazardous items always name the exact material (for example paint thinner, fluorescent tube, medicine) and start the advice with a clear safety warning.
4. Mention visible contamination (food, grease, liquids, mixed materials) and how it changes disposal.
5. optimize_recycling is called ONCE, in that same step, with every item, passing each item's id exactly as analyze_image gave it (name, final category, contamination, weight_g=0 unless the user stated it, and tip = one specific tip of at most 20 words for disposing of, reusing or decomposing THAT item more efficiently and with less environmental harm, for example how to compost it, what to separate first, or how to reuse it). Its numbers are the optimization result: never compute or invent your own.

Reply format. The app already shows per-item cards (category, breakdown time, what to avoid), the bin list and the optimization numbers, so do NOT repeat them. Write one short lead sentence, then a numbered list of at most 5 practical steps, most important first, each naming the bin it ends in. For hazardous items put the safety warning first. Mention any override of the CNN in one short clause. Stay under about 110 words in total. Local rules vary: say so once. Never invent tool results. For follow-up questions, answer from the conversation; only re-run tools for a new image.

Finish EVERY reply (including greetings and general questions) with one last line in exactly this form:
SUGGESTIONS: <question 1> | <question 2> | <question 3>
The three questions are what this user is most likely to ask next, specific to this conversation, written the way the user would say them, under 9 words each. Never repeat a suggestion you already gave."""


class Item(BaseModel):
    name: str
    box_2d: list[int]
    category: CATEGORIES
    contamination: Literal["none", "light", "heavy"]


class ChatSession:
    def __init__(self):
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(timeout=CALL_TIMEOUT_MS))
        self._crops, self._crops_for = None, None
        self.model_idx = 0  # position in MODELS; moves forward when a model is slow, overloaded or out of quota
        self.image_path: str | None = None
        self.calls: list[dict] = []
        self.cams: dict[int, str | None] = {}
        self.on_event = None  # optional callback for live progress (used by the streaming endpoint)
        self._config = types.GenerateContentConfig(
            system_instruction=SYSTEM,
            tools=[self.analyze_image, self.get_disposal_guide, self.optimize_recycling],
            temperature=0.3,
        )
        self.chat_idx = 0
        self.chat = self.client.chats.create(model=MODELS[0], config=self._config)

    def _chat_for(self, idx: int, history) -> None:
        """Make sure the chat object talks to MODELS[idx]; on a switch it restarts from `history` (the conversation before this turn)."""
        if idx != self.chat_idx:
            self.chat = self.client.chats.create(model=MODELS[idx], config=self._config, history=history)
            self.chat_idx = idx

    def _generate(self, **kwargs):
        """One-shot generate_content that falls through the model ladder on slow or failing models."""
        for i in range(self.model_idx, len(MODELS)):
            try:
                resp = self.client.models.generate_content(model=MODELS[i], **kwargs)
                self.model_idx = i
                return resp
            except Exception as e:
                if not _retriable(e) or i == len(MODELS) - 1:
                    raise

    # ---- tools (docstrings are what Gemini reads) ----
    def analyze_image(self) -> dict:
        """Analyze the most recently uploaded photo. Detects each distinct waste item, crops it, and classifies each crop with the trained CNN. Returns a list of items with name, cnn_category, cnn_confidence (0-1) and top_k candidates."""
        if not self.image_path:
            return {"error": "no image uploaded yet"}
        self._emit("detect", "Detecting items", "start")
        if self._crops_for != self.image_path:  # a model fallback replays the turn; don't pay for detection twice
            self._crops, self._crops_for = self._detect_and_crop(), self.image_path
        crops = self._crops
        self._emit("detect", "Detecting items", "done", f"{len(crops)} found")
        self._emit("classify", "Classifying with ResNet50", "start")
        items = []
        for i, (name, crop, box, gcat, gcon) in enumerate(crops, 1):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                crop.save(tmp, "JPEG")
            try:
                r = classifier.classify(tmp.name, cam=True)
            finally:
                Path(tmp.name).unlink(missing_ok=True)
            self.cams[i] = r.pop("cam", None)  # heatmap goes to the UI only, never to the model
            items.append({"id": i, "name": name, "box": box, "cnn_category": r["category"],
                          "cnn_confidence": r["confidence"], "top_k": r["top_k"],
                          "gemini_category": gcat, "gemini_contamination": gcon,
                          "suggested_category": self._combine(r["category"], gcat)})
        self._emit("classify", "Classifying with ResNet50", "done", f"{len(items)} checked")
        self.calls.append({"tool": "analyze_image", "result": items})
        return {"items": items}

    def get_disposal_guide(self, category: str) -> dict:
        """Get the correct bin, disposal steps, decomposition time and what to avoid for a waste category. category must be one of: cardboard, glass, metal, paper, plastic, trash, organic, battery, e-waste, textile, hazardous."""
        self._emit(f"guide-{category}", f"Looking up {category} disposal", "start")
        self.calls.append({"tool": "get_disposal_guide", "args": {"category": category}})
        self._emit(f"guide-{category}", f"Looking up {category} disposal", "done")
        return knowledge.GUIDES.get(category.lower().strip(), {"error": f"unknown category {category}"})

    def optimize_recycling(self, items: list[optimizer.PlanItem]) -> dict:
        """Build the optimized recycling plan for ALL items found in the photo. For each item give its name, final category, contamination (none, light = food residue or liquid, heavy = grease-soaked, wet or mixed materials) and weight_g (0 unless the user stated the weight). Returns items grouped by bin, the recovered share of waste as-is vs after following the prep steps, CO2 avoided in both cases, and the prep actions."""
        self._emit("optimize", "Optimizing the recycling plan", "start")
        result = optimizer.optimize(items)
        self.calls.append({"tool": "optimize_recycling", "result": result})
        self._emit("optimize", "Optimizing the recycling plan", "done", f"{result['recovered_share_after_prep_pct']}% recovered after prep")
        return result

    # ---- helpers ----
    def _emit(self, step: str, label: str, state: str, detail: str = "") -> None:
        if self.on_event:
            self.on_event({"type": "step", "id": step, "label": label, "state": state, "detail": detail})

    @staticmethod
    def _combine(cnn_category: str, gemini_category: str) -> str:
        """Gemini sees the whole scene and knows all 11 categories; the CNN only knows 6 clean-photo materials. Trust Gemini, fall back to the CNN."""
        return gemini_category or cnn_category

    def _detect_and_crop(self):
        full = Image.open(self.image_path).convert("RGB")
        try:
            resp = self._generate(
                contents=[self._part(self.image_path),
                          f"Detect every distinct waste item in this photo (at most {MAX_ITEMS}), including small objects (pens, cables), large items (boxes, bags, containers) and food leftovers. "
                          "For each item return: a short specific name; box_2d as [ymin, xmin, ymax, xmax] normalised to 0-1000; its category; "
                          "and contamination (none, light = food or liquid residue, heavy = grease-soaked, wet or mixed materials). " + MATERIAL_HINTS],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", response_schema=list[Item], temperature=0),
            )
            detected = [Item(**d) for d in json.loads(resp.text)][:MAX_ITEMS]
        except Exception:
            detected = []
        w, h = full.size
        out = []
        for it in detected:
            if len(it.box_2d) != 4:
                continue
            y0, x0, y1, x1 = (max(0, min(1000, v)) for v in it.box_2d)
            pad_x, pad_y = (x1 - x0) * 0.05, (y1 - y0) * 0.05
            box = (int((x0 - pad_x) / 1000 * w), int((y0 - pad_y) / 1000 * h),
                   int((x1 + pad_x) / 1000 * w), int((y1 + pad_y) / 1000 * h))
            box = (max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3]))
            if box[2] - box[0] >= 16 and box[3] - box[1] >= 16:
                out.append((it.name, full.crop(box), [round(x0 / 1000, 3), round(y0 / 1000, 3), round(x1 / 1000, 3), round(y1 / 1000, 3)], it.category, it.contamination))
        return out or [("whole image", full, [0, 0, 1, 1], "", "none")]

    @staticmethod
    def _part(path: str) -> types.Part:
        img = Image.open(path).convert("RGB")
        img.thumbnail((1024, 1024))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")

    def send(self, message: str, image_path: str | None = None) -> dict:
        self.calls = []
        self.cams = {}
        parts = []
        if image_path:
            if self.image_path:
                Path(self.image_path).unlink(missing_ok=True)
            self.image_path = image_path
            parts.append(self._part(image_path))
            message = message or "Please analyze this."
        parts.append(message)
        response = None
        base_history = list(self.chat.get_history())
        for i in range(self.model_idx, len(MODELS)):
            try:
                self._chat_for(i, base_history)
                response = self.chat.send_message(parts)
                if not (response.text or "").strip():
                    raise RuntimeError("empty response")
                self.model_idx = i
                break
            except Exception as e:
                if not _retriable(e) or i == len(MODELS) - 1:
                    raise
                reason = str(e).replace("\n", " ")[:70] or type(e).__name__
                print(f"[agent] {MODELS[i]} failed ({reason}); trying {MODELS[i + 1]}", flush=True)
                self._emit("fallback", f"{MODELS[i]} was slow or unavailable, trying {MODELS[i + 1]}", "done", reason[:40])
                self.calls, self.cams = [], {}  # the next model redoes the tool calls from scratch
        answer, suggestions = self._split_suggestions(response.text or "")
        return {"answer": answer, "suggestions": suggestions, "tool_calls": self.calls, "model": MODELS[self.model_idx], **self._structured()}

    @staticmethod
    def _split_suggestions(text: str) -> tuple[str, list[str]]:
        m = re.search(r"(?:^|\n)[ \t]*\**SUGGESTIONS:?\**[ \t]*(.+?)\s*$", text, re.S | re.I)
        if not m:
            return text.strip(), []
        picks = [q.strip(" *_-\"'") for q in m.group(1).replace("\n", " ").split("|")]
        return text[:m.start()].strip(), [q for q in picks if 3 < len(q) <= 70][:3]

    def _structured(self) -> dict:
        """Join detection, CNN and optimizer results for the UI cards."""
        analyzed = next((c["result"] for c in self.calls if c["tool"] == "analyze_image"), [])
        plan = next((c["result"] for c in self.calls if c["tool"] == "optimize_recycling"), None)
        plan_rows = (plan or {}).get("items", [])
        by_id = {r["id"]: r for r in plan_rows if r.get("id")}
        by_name: dict[str, list] = {}
        for r in plan_rows:
            by_name.setdefault(r["item"].lower(), []).append(r)
        items = []
        for idx, it in enumerate(analyzed):
            # the agent may rename an item, so match by id first, then name, then position
            row = by_id.get(it["id"])
            if row is None and by_name.get(it["name"].lower()):
                row = by_name[it["name"].lower()].pop(0)
            if row is None and len(plan_rows) == len(analyzed):
                row = plan_rows[idx]
            cat = row["category"] if row else it.get("suggested_category") or it["cnn_category"]
            takes, avoid = knowledge.SHORT.get(cat, ("", ""))
            items.append({"name": it["name"], "box": it["box"], "cnn_category": it["cnn_category"],
                          "cnn_confidence": it["cnn_confidence"], "category": cat,
                          "top_k": it["top_k"], "cam": self.cams.get(it["id"]),
                          "contamination": row["contamination"] if row else it.get("gemini_contamination", "none"),
                          "tip": row.get("tip", "") if row else "", "gemini_category": it.get("gemini_category", ""),
                          "takes": takes, "avoid": avoid})
        bins: dict[str, list[int]] = {}
        for idx, it in enumerate(items):
            bins.setdefault(knowledge.GUIDES.get(it["category"], {}).get("bin", "General waste"), []).append(idx)
        return {"items": items, "bins": bins, "optimization": plan}
