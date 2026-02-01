from ultralytics import YOLO
import mss
import pygetwindow as gw
import numpy as np
import cv2
from utils.MoveMouse import move_mouse
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController
import time
import random

# =======================
# CONFIG
# =======================
model_path = 'versions/v1.pt'
window_title = 'Counter-Strike 2'
WIDTH, HEIGHT = 1280, 720
INFERENCE_SIZE = (640, 384)

# =======================
# INIT
# =======================
model = YOLO(model_path)
mouse = MouseController()
keyboard = KeyboardController()

cv2.namedWindow("YOLO CS", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO CS", WIDTH, HEIGHT)

# =======================
# HELPERS
# =======================
def screenshot(window_title):
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        raise Exception(f"Window not found: {window_title}")
    win = windows[0]

    monitor = {
        "top": win.top,
        "left": win.left,
        "width": win.width,
        "height": win.height
    }

    with mss.mss() as sct:
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def detect(model, img):
    resized = cv2.resize(img, INFERENCE_SIZE)
    results = model(resized)
    annotated = results[0].plot()
    return results, annotated

class TriggerBot:
    def __init__(self, target_side='ct', screen_width=1280, screen_height=720):
        self.target_side = target_side
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_center = np.array([screen_width // 2, screen_height // 2])

        if target_side == 'ct':
            self.target_classes = [0, 1]
        else:
            self.target_classes = [2, 3]

    def is_center_inside_target(self, results):
        boxes = results[0].boxes
        for box, cls in zip(boxes.xyxy, boxes.cls):
            if int(cls) not in self.target_classes:
                continue
            x1, y1, x2, y2 = box.cpu().numpy()
            # масштаб назад до оригінального вікна
            x1 = x1 * (WIDTH / INFERENCE_SIZE[0])
            x2 = x2 * (WIDTH / INFERENCE_SIZE[0])
            y1 = y1 * (HEIGHT / INFERENCE_SIZE[1])
            y2 = y2 * (HEIGHT / INFERENCE_SIZE[1])

            if x1 <= self.screen_center[0] <= x2 and y1 <= self.screen_center[1] <= y2:
                return True, {'box': (x1, y1, x2, y2), 'class': int(cls)}
        return False, None

bot = TriggerBot(target_side='ct')

# =======================
# MAIN LOOP
# =======================
while True:
    try:
        frame = screenshot(window_title)
        results, annotated_frame = detect(model, frame)
        cv2.imshow("YOLO CS", annotated_frame)
        triggered, target = bot.is_center_inside_target(results)
        ctrl_choice = [0, 1]
        ctrl_weigths = [0.6, 0.4]
        if triggered:
            cls = target['class']
            ctrl_status = random.choices(ctrl_choice, weights=ctrl_weigths)[0]

            if cls == 0:
                if ctrl_status == 1:
                    keyboard.press(Key.ctrl)
                    mouse.press(Button.left)
                    time.sleep(0.5)
                    mouse.release(Button.left)
                    keyboard.release(Key.ctrl)
                else:
                    mouse.press(Button.left)
                    time.sleep(0.4)
                    mouse.release(Button.left)
                    time.sleep(0.2)

            elif cls == 1:
                mouse.press(Button.left)
                mouse.release(Button.left)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    except Exception:
        pass

cv2.destroyAllWindows()
