"""Home page for the HMI."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QRect, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

HOME_ICON_FILES = {
    "quick": (
        "quickmove_24dp_000000_FILL0_wght400_GRAD0_opsz24.svg",
        "quickmove_24dp_FFFFFF_FILL0_wght400_GRAD0_opsz24.svg",
    ),
    "job": (
        "Job_24dp_000000_FILL0_wght400_GRAD0_opsz24.svg",
        "Job_24dp_FFFFFF_FILL0_wght400_GRAD0_opsz24.svg",
    ),
    "settings": (
        "settings_24dp_000000_FILL0_wght400_GRAD0_opsz24.svg",
        "settings_24dp_FFFFFF_FILL0_wght400_GRAD0_opsz24.svg",
    ),
}


class HomePage(QWidget):
    open_page_requested = pyqtSignal(int)
    connect_clicked = pyqtSignal()
    disconnect_clicked = pyqtSignal()
    home_clicked = pyqtSignal()
    camera_clicked = pyqtSignal()

    PAGE_QUICK_MOVE = 1
    PAGE_JOB = 2
    PAGE_SETTINGS = 3

    def __init__(self):
        super().__init__()
        self._current_theme = "light"
        self._bg_light_pixmap = None
        self._bg_dark_pixmap = None
        self._bg_pixmap = None
        self._tile_icon_map = {}
        self._build_ui()
        self._load_theme_backgrounds()
        self.set_theme("light")

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(14)

        self.left_panel = QFrame()
        self.left_panel.setObjectName("homeLeftPanel")
        self.left_panel.setAttribute(Qt.WA_StyledBackground, True)
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(18, 18, 18, 18)
        left_layout.setSpacing(12)

        self.quick_move_tile = QPushButton("Quick Move")
        self.quick_move_tile.setObjectName("tileBtn")
        self.quick_move_tile.setMinimumHeight(84)
        self.quick_move_tile.clicked.connect(lambda: self.open_page_requested.emit(self.PAGE_QUICK_MOVE))

        self.job_tile = QPushButton("Job")
        self.job_tile.setObjectName("tileBtn")
        self.job_tile.setMinimumHeight(84)
        self.job_tile.clicked.connect(lambda: self.open_page_requested.emit(self.PAGE_JOB))

        self.settings_tile = QPushButton("Settings")
        self.settings_tile.setObjectName("tileBtn")
        self.settings_tile.setMinimumHeight(84)
        self.settings_tile.clicked.connect(lambda: self.open_page_requested.emit(self.PAGE_SETTINGS))

        self._menu_buttons = [self.quick_move_tile, self.job_tile, self.settings_tile]

        left_layout.addWidget(self.quick_move_tile)
        left_layout.addWidget(self.job_tile)
        left_layout.addWidget(self.settings_tile)
        left_layout.addStretch()

        root.addWidget(self.left_panel, stretch=2)
        root.addStretch(3)

    def _load_theme_backgrounds(self):
        pic_dir = Path(__file__).resolve().parents[2] / "Picture"

        light_path = pic_dir / "moceai.png"
        self._bg_light_pixmap = QPixmap(str(light_path)) if light_path.exists() else None
        if self._bg_light_pixmap is not None and self._bg_light_pixmap.isNull():
            self._bg_light_pixmap = None

        dark_candidates = [
            pic_dir / "moceai_Drak.jpg",
            pic_dir / "moceai_Dark.jpg",
            pic_dir / "moceai_dark.jpg",
        ]
        self._bg_dark_pixmap = None
        for p in dark_candidates:
            if not p.exists():
                continue
            px = QPixmap(str(p))
            if px.isNull():
                continue
            self._bg_dark_pixmap = px
            break

        self._tile_icon_map = {}
        for name, (light_file, dark_file) in HOME_ICON_FILES.items():
            light_icon = QIcon(str(pic_dir / light_file)) if (pic_dir / light_file).exists() else QIcon()
            dark_icon = QIcon(str(pic_dir / dark_file)) if (pic_dir / dark_file).exists() else QIcon()
            self._tile_icon_map[name] = {"light": light_icon, "dark": dark_icon}

    def set_theme(self, theme: str):
        self._current_theme = "dark" if str(theme).strip().lower() == "dark" else "light"
        if self._current_theme == "dark":
            self._bg_pixmap = self._bg_dark_pixmap or self._bg_light_pixmap
        else:
            self._bg_pixmap = self._bg_light_pixmap or self._bg_dark_pixmap
        self._apply_tile_icons()
        self._apply_overlay_styles()
        self.update()

    def _apply_tile_icons(self):
        icon_buttons = {
            "quick": self.quick_move_tile,
            "job": self.job_tile,
            "settings": self.settings_tile,
        }
        for key, btn in icon_buttons.items():
            pair = self._tile_icon_map.get(key, {})
            icon = pair.get(self._current_theme)
            if icon is None or icon.isNull():
                icon = pair.get("light")
            if icon is None or icon.isNull():
                btn.setIcon(QIcon())
                continue
            btn.setIcon(icon)
            btn.setIconSize(QSize(18, 18))

    def _apply_overlay_styles(self):
        if self._current_theme == "dark":
            btn_bg = "rgba(12, 20, 32, 0.38)"
            btn_hover = "rgba(20, 34, 52, 0.52)"
            btn_press = "rgba(12, 20, 32, 0.62)"
            border = "rgba(255, 255, 255, 0.20)"
            text = "#ECF2FA"
        else:
            btn_bg = "rgba(255, 255, 255, 0.32)"
            btn_hover = "rgba(255, 255, 255, 0.52)"
            btn_press = "rgba(244, 248, 252, 0.70)"
            border = "rgba(0, 0, 0, 0.14)"
            text = "#1F2733"

        self.left_panel.setStyleSheet(
            f"""
            QFrame#homeLeftPanel {{
                background: transparent;
                border: none;
            }}
            QFrame#homeLeftPanel QPushButton {{
                border-radius: 14px;
                border: 1px solid {border};
                background: {btn_bg};
                color: {text};
                text-align: left;
                padding-left: 16px;
                font-weight: 600;
            }}
            QFrame#homeLeftPanel QPushButton:hover {{
                background: {btn_hover};
            }}
            QFrame#homeLeftPanel QPushButton:pressed {{
                background: {btn_press};
            }}
            """
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        if self._bg_pixmap is not None and not self._bg_pixmap.isNull():
            target = self.rect()
            scaled = self._bg_pixmap.scaled(
                target.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
            sx = max(0, (scaled.width() - target.width()) // 2)
            sy = max(0, (scaled.height() - target.height()) // 2)
            source = QRect(sx, sy, target.width(), target.height())
            painter.setOpacity(0.9 if self._current_theme == "dark" else 0.8)
            painter.drawPixmap(target, scaled, source)
            painter.setOpacity(1.0)
        painter.end()

    def set_connected(self, connected: bool):
        _ = connected

    def set_texts(self, tr):
        self.quick_move_tile.setText(tr("home_tile_quick_move"))
        self.job_tile.setText(tr("home_tile_job"))
        self.settings_tile.setText(tr("home_tile_settings"))
