"""Standalone camera tool window."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


class CameraWindow(QMainWindow):
    """Independent camera tool window with fullscreen/topmost controls."""

    closed = pyqtSignal()

    def __init__(self, title: str, fps_text: str, latency_text: str):
        super().__init__(None)
        self.setWindowTitle(title)
        self.setMinimumSize(800, 600)
        self.resize(1080, 760)

        self.overlay_visible = True
        self._always_on_top = False
        self._rtt_text = "--"

        self._build_toolbar()
        self._build_body(fps_text, latency_text)

    def _build_toolbar(self):
        toolbar = QToolBar("CameraToolbar", self)
        toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        self.fullscreen_action = QAction("⛶", self)
        self.fullscreen_action.setCheckable(True)
        self.fullscreen_action.triggered.connect(self.toggle_fullscreen)
        toolbar.addAction(self.fullscreen_action)

        self.topmost_action = QAction("📌", self)
        self.topmost_action.setCheckable(True)
        self.topmost_action.triggered.connect(self.toggle_topmost)
        toolbar.addAction(self.topmost_action)

        self.overlay_action = QAction("HUD", self)
        self.overlay_action.setCheckable(True)
        self.overlay_action.setChecked(True)
        self.overlay_action.triggered.connect(self.toggle_overlay)
        toolbar.addAction(self.overlay_action)

    def _build_body(self, fps_text: str, latency_text: str):
        body = QWidget()
        self.setCentralWidget(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(8, 8, 8, 8)

        self.camera_label = QLabel()
        self.camera_label.setObjectName("cameraLabel")
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.camera_label, stretch=1)

        footer = QHBoxLayout()
        self.fps_label = QLabel(fps_text)
        self.latency_label = QLabel(latency_text)
        self.rtt_label = QLabel("RTT: --")
        self.rec_label = QLabel("")
        self.rec_label.setObjectName("recLabel")

        footer.addWidget(self.fps_label)
        footer.addSpacing(10)
        footer.addWidget(self.latency_label)
        footer.addSpacing(10)
        footer.addWidget(self.rtt_label)
        footer.addStretch()
        footer.addWidget(self.rec_label)

        layout.addLayout(footer)

    def set_window_title(self, title: str):
        self.setWindowTitle(title)

    def set_overlay_visible(self, visible: bool):
        self.overlay_visible = bool(visible)
        self.fps_label.setVisible(self.overlay_visible)
        self.latency_label.setVisible(self.overlay_visible)
        self.rtt_label.setVisible(self.overlay_visible)
        self.rec_label.setVisible(self.overlay_visible)

    def toggle_overlay(self):
        self.set_overlay_visible(self.overlay_action.isChecked())

    def toggle_fullscreen(self):
        if self.fullscreen_action.isChecked():
            self.showFullScreen()
        else:
            self.showNormal()

    def toggle_topmost(self):
        self._always_on_top = self.topmost_action.isChecked()
        self.setWindowFlag(Qt.WindowStaysOnTopHint, self._always_on_top)
        self.show()
        self.raise_()

    def set_placeholder(self, pixmap: QPixmap | None, text: str):
        if pixmap is not None:
            scaled = pixmap.scaled(
                self.camera_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.camera_label.setPixmap(scaled)
            self.camera_label.setText("")
        else:
            self.camera_label.setPixmap(QPixmap())
            self.camera_label.setText(text)

    def set_frame(self, frame: np.ndarray, fps_text: str, latency_text: str, rtt_text: str, recording: bool):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        scaled = qimg.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_label.setPixmap(QPixmap.fromImage(scaled))
        self.camera_label.setText("")

        self.fps_label.setText(fps_text)
        self.latency_label.setText(latency_text)
        self.rtt_label.setText(rtt_text)
        self.rec_label.setText("● REC" if recording else "")

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)
