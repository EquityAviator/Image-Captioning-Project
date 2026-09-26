"""
Experiment A - Offline knowledge distillation: generate BLIP pseudo-captions
for all training images (5 samples each), then retrain Gen-3 CE on
original + synthetic captions. BLIP is used OFFLINE only; inference stays
purely local (CLIP + LSTM).

Output: weights/blip_pseudo_captions.json  {image: [cap1..cap5]}
"""
import json, time
from pathlib import Path

import torch
from PIL import Image

BACKEND = Path(__file__).resolve().parent
PROJECT = BACKEND.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import BlipForConditionalGeneration, BlipProcessor

proc = BlipProcessor.from_pretrained(str(PROJECT / "weights/blip"),
                                     local_files_only=True)
model = BlipForConditionalGeneration.from_pretrained(
    str(PROJECT / "weights/blip"), local_files_only=True).to(DEVICE).eval()

with open(PROJECT / "weights/train_images.json") as fh:
    train_images = json.load(fh)

out_path = PROJECT / "weights/blip_pseudo_captions.json"
partial = {}
if out_path.exists():
    partial = json.loads(out_path.read_text())
    print(f">> resuming: {len(partial)} images already done", flush=True)

t0 = time.time()
done = len(partial)
torch.manual_seed(42)

for n, img in enumerate(train_images):
    if img in partial:
        continue
    image = Image.open(PROJECT / "dataset/Images" / img).convert("RGB")
    inputs = proc(image, return_tensors="pt").to(DEVICE)
    caps = set()
    with torch.no_grad():
        # beam-ish deterministic caption first
        out = model.generate(**inputs, max_length=40, num_beams=3)
        caps.add(proc.decode(out[0], skip_special_tokens=True).strip())
        # 4 sampled captions for diversity
        for _ in range(4):
            out = model.generate(**inputs, max_length=40,
                                 do_sample=True, top_p=0.9, temperature=1.0)
            caps.add(proc.decode(out[0], skip_special_tokens=True).strip())
    partial[img] = sorted(caps)
    done += 1
    if done % 250 == 0:
        out_path.write_text(json.dumps(partial))
        el = time.time() - t0
        print(f"  {done}/{len(train_images)} ({el:.0f}s, "
              f"{el/max(done-len(partial) if False else done,1):.2f}s/img cum)",
              flush=True)

out_path.write_text(json.dumps(partial))
print(f">> done: {len(partial)} images, saved {out_path}")
