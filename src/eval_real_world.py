"""Evaluate on the out-of-dataset photos in data/real_world.

Folder name = true class. Run:
  python -m src.eval_real_world            # CNN only (no API calls)
  python -m src.eval_real_world --agent    # also run the full Gemini agent on every image
"""
import argparse
import csv
import json
from pathlib import Path

from . import classifier

ROOT = Path(__file__).resolve().parent.parent / "data" / "real_world"
CNN_CLASSES = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
IMG = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", action="store_true")
    ap.add_argument("--only", nargs="*", help="limit the agent run to these folders")
    args = ap.parse_args()

    images = [(d.name, p) for d in sorted(ROOT.iterdir()) if d.is_dir() for p in sorted(d.iterdir()) if p.suffix.lower() in IMG]
    rows = []
    for true, path in images:
        r = classifier.classify(str(path))
        rows.append({"file": f"{true}/{path.name}", "true": true, "cnn": r["category"], "cnn_conf": r["confidence"]})

    scored = [r for r in rows if r["true"] in CNN_CLASSES]
    ok = sum(r["true"] == r["cnn"] for r in scored)
    print(f"CNN alone, in-scope classes: {ok}/{len(scored)} = {ok / max(len(scored), 1):.1%}")
    for r in scored:
        print(f"  {'OK ' if r['true'] == r['cnn'] else 'BAD'} {r['file']:<22} -> {r['cnn']} ({r['cnn_conf']})")
    print("CNN on items it was never trained for (expected to be wrong):")
    for r in rows:
        if r["true"] not in CNN_CLASSES and r["true"] != "multi_item":
            print(f"  {r['file']:<22} -> {r['cnn']} ({r['cnn_conf']})")

    if args.agent:
        from .agent import ChatSession
        for r in rows:
            folder = r["file"].split("/")[0]
            if args.only and folder not in args.only:
                continue
            try:
                out = ChatSession().send("sort this", str(ROOT / r["file"]))
            except Exception as e:
                r["agent"] = f"ERROR {str(e)[:80]}"
                print("agent error:", r["file"], r["agent"])
                if "429" in str(e):
                    break
                continue
            cats = []
            for c in out["tool_calls"]:
                if c["tool"] == "optimize_recycling":
                    cats = [i["category"] for i in c["result"]["items"]]
            r["agent"] = cats
            print(f"  agent {r['file']:<22} true={r['true']:<10} -> {cats}")
        agent_rows = [r for r in rows if isinstance(r.get("agent"), list) and r["true"] != "multi_item"]
        good = sum(r["true"] in r["agent"] and len(set(r["agent"])) == 1 for r in agent_rows)
        print(f"Agent final category correct: {good}/{len(agent_rows)} = {good / max(len(agent_rows), 1):.1%}")

    (ROOT / "results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
