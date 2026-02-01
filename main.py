from ultralytics import YOLO
import mss, pygetwindow as gw, numpy as np, cv2
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController, Listener as KeyListener, KeyCode
import time, random, threading
import torch
from utils.MoveMouse import move_mouse
from PyQt5.QtWidgets import QApplication
from ui_monitor import MonitorUI

# =======================
# CONFIG
# =======================
model_path = 'versions/v1.pt'
window_title = 'Counter-Strike 2'
WIDTH, HEIGHT = 1280, 720

runtime_config = {
    "inference_size": (320, 192),
    "show_preview": True,
    "mode": "balanced",
    "conf_threshold": 0.4,
    "trigger_enabled": True,
    "smooth_aiming": True,
    "smooth_steps": 24,
    "smooth_noise": 0.1,
    "smooth_delay": 0.001
}

# =======================
# INIT
# =======================
torch.backends.cudnn.benchmark = True

model = YOLO(model_path)
if torch.cuda.is_available():
    model = model.cuda().half()

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
    results = model(resized, conf=cfg["conf_threshold"], verbose=False, device='cuda' if torch.cuda.is_available() else 'cpu')
    inf_ms = (time.time() - t0) * 1000
    annotated = results[0].plot()
    return results, annotated, inf_ms

# =======================
# TRIGGER BOT
# =======================
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
            x1 = x1 * (self.screen_width / runtime_config["inference_size"][0])
            x2 = x2 * (self.screen_width / runtime_config["inference_size"][0])
            y1 = y1 * (self.screen_height / runtime_config["inference_size"][1])
            y2 = y2 * (self.screen_height / runtime_config["inference_size"][1])
            if x1 <= self.screen_center[0] <= x2 and y1 <= self.screen_center[1] <= y2:
                return True, {'box': (x1, y1, x2, y2), 'class': int(cls)}
        return False, None

bot = TriggerBot(target_side='ct')

# =======================
# AIM ASSIST
# =======================
class AimAssist:
    def __init__(self, target_side='ct', target_part='head', screen_width=WIDTH, screen_height=HEIGHT):
        self.target_side = target_side
        self.target_part = target_part
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_center = np.array([screen_width // 2, screen_height // 2])
        self.active = False

        if target_side == 'ct':
            self.target_classes = [0] if target_part=='body' else [1]
        else:
            self.target_classes = [2] if target_part=='body' else [3]

        self.listener = KeyListener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        if key == KeyCode.from_char('c'):
            self.active = True

    def on_release(self, key):
        if key == KeyCode.from_char('c'):
            self.active = False

    def get_closest_target(self, results):
        boxes = results[0].boxes
        closest = None
        min_dist = float('inf')
        for box, cls in zip(boxes.xyxy, boxes.cls):
            cls = int(cls)
            if cls not in self.target_classes:
                continue
            x1, y1, x2, y2 = box.cpu().numpy()
            x1 = x1 * (self.screen_width / runtime_config["inference_size"][0])
            x2 = x2 * (self.screen_width / runtime_config["inference_size"][0])
            y1 = y1 * (self.screen_height / runtime_config["inference_size"][1])
            y2 = y2 * (self.screen_height / runtime_config["inference_size"][1])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            dist = np.linalg.norm(np.array([cx, cy]) - self.screen_center)
            if dist < min_dist:
                min_dist = dist
                closest = (cx, cy)
        return closest

    def aim_at_target(self, target_pos):
        if target_pos is None:
            return


        center_x, center_y = self.screen_center
        target_x, target_y = target_pos

        dx_total = target_x - center_x
        dy_total = target_y - center_y

        steps = runtime_config.get("smooth_steps", 16)
        step_dx = dx_total / steps
        step_dy = dy_total / steps

        for _ in range(steps):

            noise_x = random.gauss(0, runtime_config.get("smooth_noise", 0.2))
            noise_y = random.gauss(0, runtime_config.get("smooth_noise", 0.2))

            move_mouse(int(step_dx + noise_x), int(step_dy + noise_y))

            time.sleep(runtime_config.get("smooth_delay", 0.01) * random.uniform(0.9, 1.1))

aim_assist = AimAssist(target_side='ct', target_part='head')

# =======================
# MAIN LOOP
# =======================
def main_loop():
    prev_time = time.time()
    while True:
        try:
            frame = screenshot(window_title)
            results, annotated_frame, inf_ms = detect(model, frame, runtime_config)

            ui.update_frame(annotated_frame)

            now = time.time()
            fps = 1 / (now - prev_time) if (now - prev_time) > 0 else 0
            prev_time = now

            boxes = results[0].boxes
            dets = len(boxes)
            avg_conf = float(boxes.conf.mean()) if dets else 0
            ui.update_perf(fps, inf_ms, avg_conf, dets)

            # TRIGGER BOT
            if runtime_config.get("trigger_enabled", True):
                triggered, target = bot.is_center_inside_target(results)
                if triggered:
                    cls = target['class']
                    ctrl_choice = [0, 1]
                    ctrl_weights = [0.6, 0.4]
                    ctrl_status = random.choices(ctrl_choice, weights=ctrl_weights)[0]
                    x1, y1, x2, y2 = target['box']
                    target_x = (x1 + x2) / 2
                    target_y = (y1 + y2) / 2
                    if cls == 0:
                        if ctrl_status == 1:
                            keyboard.press(Key.ctrl)
                            mouse.press(Button.left)
                            time.sleep(0.3)
                            mouse.release(Button.left)
                            keyboard.release(Key.ctrl)
                        else:
                            mouse.press(Button.left)
                            time.sleep(0.2)
                            mouse.release(Button.left)
                    elif cls == 1:
                        mouse.press(Button.left)
                        time.sleep(0.01)
                        mouse.release(Button.left)

            if aim_assist.active:
                target_pos = aim_assist.get_closest_target(results)
                aim_assist.aim_at_target(target_pos)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.5)

# =======================
# ENTRY POINT
# =======================
if __name__ == "__main__":
    app = QApplication([])
    ui = MonitorUI(runtime_config)
    ui.show()

    threading.Thread(target=main_loop, daemon=True).start()
    app.exec_()
