from ultralytics import YOLO
import mss, pygetwindow as gw, numpy as np, cv2
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController
import time, random, threading

from PyQt5.QtWidgets import QApplication
from ui_monitor import MonitorUI

# =======================
# CONFIG
# =======================
model_path = 'versions/v1.pt'
window_title = 'Counter-Strike 2'
WIDTH, HEIGHT = 1280, 720

runtime_config = {
    "inference_size": (640, 384),
    "show_preview": True,
    "mode": "balanced",
    "conf_threshold": 0.4
}

# =======================
# INIT
# =======================
model = YOLO(model_path)
mouse = MouseController()
keyboard = KeyboardController()

# =======================
# HELPERS
# =======================
def screenshot(window_title):
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        raise Exception(f"Window not found: {window_title}")
    win = windows[0]
    monitor = {"top": win.top, "left": win.left, "width": win.width, "height": win.height}
    with mss.mss() as sct:
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def detect(model, img, cfg):
    resized = cv2.resize(img, cfg["inference_size"])
    t0 = time.time()
    results = model(resized, conf=cfg["conf_threshold"])
    inf_ms = (time.time() - t0) * 1000
    annotated = results[0].plot()
    return results, annotated, inf_ms

class TriggerBot:
    def __init__(self, target_side='ct', screen_width=WIDTH, screen_height=HEIGHT):
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
            x1 = x1 * (self.screen_width / runtime_config["inference_size"][0])
            x2 = x2 * (self.screen_width / runtime_config["inference_size"][0])
            y1 = y1 * (self.screen_height / runtime_config["inference_size"][1])
            y2 = y2 * (self.screen_height / runtime_config["inference_size"][1])
            if x1 <= self.screen_center[0] <= x2 and y1 <= self.screen_center[1] <= y2:
                return True, {'box': (x1, y1, x2, y2), 'class': int(cls)}
        return False, None

bot = TriggerBot(target_side='ct')

# =======================
# MAIN LOOP
# =======================
def main_loop():
    prev_time = time.time()
    while True:
        try:
            frame = screenshot(window_title)
            results, annotated_frame, inf_ms = detect(model, frame, runtime_config)

            # update UI
            ui.update_frame(annotated_frame)

            now = time.time()
            fps = 1 / (now - prev_time)
            prev_time = now

            boxes = results[0].boxes
            dets = len(boxes)
            avg_conf = float(boxes.conf.mean()) if dets else 0
            ui.update_perf(fps, inf_ms, avg_conf, dets)

            # TriggerBot
            triggered, target = bot.is_center_inside_target(results)
            ctrl_choice = [0, 1]
            ctrl_weights = [0.6, 0.4]
            if triggered:
                cls = target['class']
                ctrl_status = random.choices(ctrl_choice, weights=ctrl_weights)[0]

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

        except Exception:
            pass

if __name__ == "__main__":
    app = QApplication([])
    ui = MonitorUI(runtime_config)
    ui.show()

    threading.Thread(target=main_loop, daemon=True).start()
    app.exec_()
