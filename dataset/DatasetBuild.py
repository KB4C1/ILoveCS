import os
import time
from mss import mss
from PIL import Image

SAVE_DIR = "dataset/images/train"
INTERVAL = 1  # секунда

os.makedirs(SAVE_DIR, exist_ok=True)

def get_next_index(folder):
    max_idx = -1
    for f in os.listdir(folder):
        if f.startswith("img_") and f.endswith(".png"):
            try:
                idx = int(f.replace("img_", "").replace(".png", ""))
                max_idx = max(max_idx, idx)
            except ValueError:
                pass
    return max_idx + 1

with mss() as sct:
    index = get_next_index(SAVE_DIR)
    print(f"Start from img_{index:04d}.png")

    while True:
        filename = f"img_{index:04d}.png"
        path = os.path.join(SAVE_DIR, filename)

        screenshot = sct.grab(sct.monitors[1])  # весь екран
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        img.save(path)

        print(f"Saved: {filename}")
        index += 1
        time.sleep(INTERVAL)
