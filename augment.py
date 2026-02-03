import os
import cv2
import albumentations as A

IN_DIR = "dataset/images/raw"
OUT_DIR = "dataset/images/raw/aug"

os.makedirs(OUT_DIR, exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.HueSaturationValue(p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.RandomGamma(p=0.2),
])

AUG_PER_IMAGE = 2  # ×4 датасет (оригінал + 2 аугмент)

for img_name in os.listdir(IN_DIR):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(IN_DIR, img_name)
    image = cv2.imread(img_path)

    if image is None:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    base, ext = os.path.splitext(img_name)

    for i in range(AUG_PER_IMAGE):
        aug = transform(image=image)["image"]

        out_name = f"{base}_{i}.png"
        out_path = os.path.join(OUT_DIR, out_name)

        cv2.imwrite(
            out_path,
            cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)
        )

print("Аугментація завершена")
