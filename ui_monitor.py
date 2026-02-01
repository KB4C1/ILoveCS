import sys, psutil, os, time
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QComboBox, QCheckBox, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer


class MonitorUI(QWidget):
    def __init__(self, runtime_config):
        super().__init__()
        self.cfg = runtime_config
        self.process = psutil.Process(os.getpid())

        self.setWindowTitle("YOLO Runtime Monitor")
        self.resize(1000, 520)

        self.image_label = QLabel()
        self.image_label.setStyleSheet("border:1px solid #333")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.last_label_size = self.image_label.size()

        self.stats = QLabel()
        self.stats.setAlignment(Qt.AlignTop)
        self.stats.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        self.mode_box = QComboBox()
        self.mode_box.addItems([
            "ultra-perfomance (212x119) - zero precision",
            "Performance (320x192)",
            "Balanced (640x384) - recommended",
            "Quality (960x544)",
            "Ultra (1280x720)",
            "FullHD (1920x1080) - may crush your PC",
            "2K (2560x1440) - heavy",
            "4K (3840x2160) - extreme"
        ])
        self.mode_box.currentTextChanged.connect(self.on_mode_change)

        self.preview_checkbox = QCheckBox("Show inference preview")
        self.preview_checkbox.setChecked(True)
        self.preview_checkbox.stateChanged.connect(
            lambda x: self.cfg.__setitem__("show_preview", bool(x))
        )

        left = QVBoxLayout()
        left.addWidget(self.image_label)

        right = QVBoxLayout()
        right.addWidget(self.stats)
        right.addWidget(QLabel("Inference mode"))
        right.addWidget(self.mode_box)
        right.addWidget(self.preview_checkbox)
        right.addStretch()

        layout = QHBoxLayout()
        layout.addLayout(left, 3)
        layout.addLayout(right, 1)
        self.setLayout(layout)
        self.apply_theme()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(500)

        self.last_fps = 0
        self.last_inf = 0
        self.avg_conf = 0
        self.dets = 0

    def update_frame(self, frame_bgr):
        if not self.cfg["show_preview"]:
            self.image_label.clear()
            return

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)

        if self.image_label.size() != self.last_label_size:
            pix_scaled = pix.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pix_scaled)
            self.last_label_size = self.image_label.size()
        else:
            self.image_label.setPixmap(pix.scaled(
                self.last_label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

    def update_perf(self, fps, inf_ms, avg_conf, dets):
        self.last_fps = fps
        self.last_inf = inf_ms
        self.avg_conf = avg_conf
        self.dets = dets

    # ==========================
    def update_stats(self):
        cpu = self.process.cpu_percent() / psutil.cpu_count()
        ram = self.process.memory_info().rss / 1024 / 1024
        self.stats.setText(
            f"""
FPS: {self.last_fps:.1f}
Inference: {self.last_inf:.2f} ms
Detections: {self.dets}
Avg confidence: {self.avg_conf:.2f}

CPU: {cpu:.1f} %
RAM: {ram:.1f} MB

Inference size: {self.cfg["inference_size"]}
"""
        )

    def on_mode_change(self, text):
        modes = {
            "ultra-perfomance (212x119) - zero precision": (212, 119),
            "Performance (320x192)": (320, 192),
            "Balanced (640x384) - recommended": (640, 384),
            "Quality (960x544)": (960, 544),
            "Ultra (1280x720)": (1280, 720),
            "FullHD (1920x1080) - may crush your PC": (1920, 1080),
            "2K (2560x1440) - heavy": (2560, 1440),
            "4K (3840x2160) - extreme": (3840, 2160)
        }
        self.cfg["inference_size"] = modes[text]
        self.cfg["mode"] = text.split("(")[0].strip().lower()

    def apply_theme(self):
        self.setStyleSheet("""
            QWidget { background:#121212; color:#eee; }
            QLabel { font-size:13px; }
            QComboBox { background:#1e1e1e; padding:4px; }
        """)
