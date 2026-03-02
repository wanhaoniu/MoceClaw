"""Job page with Recordings / Positions / Logs tabs."""

from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class JobPage(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.recordings_tab = QWidget()
        self.positions_tab = QWidget()
        self.logs_tab = QWidget()
        self.tabs.addTab(self.recordings_tab, "Recordings")
        self.tabs.addTab(self.positions_tab, "Positions")
        self.tabs.addTab(self.logs_tab, "Logs")

        self._build_recordings_tab()
        self._build_positions_tab()
        self._build_logs_tab()

    def _build_recordings_tab(self):
        layout = QVBoxLayout(self.recordings_tab)

        ctrl = QHBoxLayout()
        self.rec_name_input = QLineEdit()
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setObjectName("recordBtn")
        self.record_btn.setEnabled(False)
        ctrl.addWidget(self.rec_name_input)
        ctrl.addWidget(self.record_btn)
        layout.addLayout(ctrl)

        self.rec_list = QListWidget()
        self.rec_list.setObjectName("recList")
        layout.addWidget(self.rec_list)

        play = QHBoxLayout()
        self.play_times_label = QLabel("Play count")
        self.play_times_spin = QSpinBox()
        self.play_times_spin.setRange(1, 100)
        self.play_times_spin.setValue(1)
        self.play_btn = QPushButton("Play")
        self.play_btn.setEnabled(False)
        self.del_rec_btn = QPushButton("Delete")
        self.del_rec_btn.setEnabled(False)

        play.addWidget(self.play_times_label)
        play.addWidget(self.play_times_spin)
        play.addWidget(self.play_btn)
        play.addWidget(self.del_rec_btn)
        layout.addLayout(play)

    def _build_positions_tab(self):
        layout = QVBoxLayout(self.positions_tab)

        save = QHBoxLayout()
        self.pos_name_input = QLineEdit()
        self.save_pos_btn = QPushButton("Save Current Pose")
        self.save_pos_btn.setEnabled(False)
        save.addWidget(self.pos_name_input)
        save.addWidget(self.save_pos_btn)
        layout.addLayout(save)

        self.pos_list = QListWidget()
        layout.addWidget(self.pos_list)

        ops = QHBoxLayout()
        self.goto_btn = QPushButton("Go To")
        self.goto_btn.setEnabled(False)
        self.del_pos_btn = QPushButton("Delete")
        self.del_pos_btn.setEnabled(False)
        self.refresh_pos_btn = QPushButton("Refresh")

        ops.addWidget(self.goto_btn)
        ops.addWidget(self.del_pos_btn)
        ops.addWidget(self.refresh_pos_btn)
        layout.addLayout(ops)

    def _build_logs_tab(self):
        layout = QVBoxLayout(self.logs_tab)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

    def set_connected(self, connected: bool):
        self.record_btn.setEnabled(connected)
        self.play_btn.setEnabled(connected)
        self.del_rec_btn.setEnabled(connected)
        self.save_pos_btn.setEnabled(connected)
        self.goto_btn.setEnabled(connected)
        self.del_pos_btn.setEnabled(connected)

    def set_texts(self, tr, recording: bool):
        self.tabs.setTabText(0, tr("job_recordings"))
        self.tabs.setTabText(1, tr("job_positions"))
        self.tabs.setTabText(2, tr("job_logs"))

        self.rec_name_input.setPlaceholderText(tr("placeholder_rec_name"))
        self.record_btn.setText(tr("btn_record_stop") if recording else tr("btn_record_start"))
        self.play_times_label.setText(tr("label_play_times"))
        self.play_btn.setText(tr("btn_play"))
        self.del_rec_btn.setText(tr("btn_delete"))

        self.pos_name_input.setPlaceholderText(tr("placeholder_pos_name"))
        self.save_pos_btn.setText(tr("btn_save_pos"))
        self.goto_btn.setText(tr("btn_goto"))
        self.del_pos_btn.setText(tr("btn_delete"))
        self.refresh_pos_btn.setText(tr("btn_refresh"))
