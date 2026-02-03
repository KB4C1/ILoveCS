import os

IMAGES_DIR = "dataset/images/raw/aug"   # шлях до папки з .png

files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")])

# 1️⃣ тимчасове перейменування
for i, f in enumerate(files):
    tmp_name = f"__tmp__{i}.png"
    os.rename(
        os.path.join(IMAGES_DIR, f),
        os.path.join(IMAGES_DIR, tmp_name)
    )

start_from = 385

# 2️⃣ нормалізація імен
for i, f in enumerate(sorted(os.listdir(IMAGES_DIR))):
    new_name = f"img_{start_from + i:04d}.png"
    os.rename(
        os.path.join(IMAGES_DIR, f),
        os.path.join(IMAGES_DIR, new_name)
    )
print("Нормалізація імен завершена")