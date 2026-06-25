import sys
import time
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QComboBox, QLineEdit, QCheckBox,
    QGridLayout, QFrame, QSpinBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QIntValidator, QFont

try:
    import sounddevice as sd
    HAS_SOUND = True
except Exception:
    HAS_SOUND = False

SR = 44100

# ---------- 音色：不同频率 + 不同包络 ----------
SOUND_PRESETS = {
    "Click（清脆）": {"accent": 1800, "normal": 1200, "sub": 900,  "decay": 70,  "shape": "sine"},
    "Wood（木鱼）":  {"accent": 1100, "normal": 750,  "sub": 550,  "decay": 90,  "shape": "tri"},
    "Beep（电子）":  {"accent": 2200, "normal": 1500, "sub": 1100, "decay": 40,  "shape": "square"},
    "Soft（柔和）":  {"accent": 880,  "normal": 587,  "sub": 440,  "decay": 30,  "shape": "sine"},
    "Rim（敲边鼓）": {"accent": 2000, "normal": 1300, "sub": 900,  "decay": 130, "shape": "noise"},
}


def make_click(freq, decay=60, shape="sine", duration=0.05):
    t = np.linspace(0, duration, int(SR * duration), False)
    if shape == "sine":
        osc = np.sin(2 * np.pi * freq * t)
    elif shape == "tri":
        osc = 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
    elif shape == "square":
        osc = np.sign(np.sin(2 * np.pi * freq * t)) * 0.7
    elif shape == "noise":
        osc = np.random.uniform(-1, 1, len(t)).astype(np.float32)
        amp = min(1.0, freq / 2000.0)   # 用 freq 当强度，区分重拍/普通/细分
        osc = osc * amp
    else:
        osc = np.sin(2 * np.pi * freq * t)
    env = np.exp(-t * decay)
    fade = min(64, len(t))
    env[:fade] *= np.linspace(0, 1, fade)
    return (osc * env).astype(np.float32)


# 预生成所有音色的三种音，避免实时回调里现合成
SOUND_CACHE = {}
for _name, _p in SOUND_PRESETS.items():
    SOUND_CACHE[_name] = {
        "accent": make_click(_p["accent"], _p["decay"], _p["shape"]),
        "normal": make_click(_p["normal"], _p["decay"], _p["shape"]),
        "sub":    make_click(_p["sub"],    _p["decay"], _p["shape"]),
    }


# ---------- 主题 ----------
THEMES = {
    "light": {
        "bg": "#FAFAFC", "text": "#1C1C1E", "sub": "#8E8E93",
        "card": "#FFFFFF", "border": "#E5E5EA", "hover": "#F2F2F7",
        "accent": "#FF3B30", "normal": "#007AFF", "idle": "#D1D1D6",
        "mute": "#E5E5EA", "blue": "#007AFF",
        "glass": "#FFFFFF", "glassBorder": "#E5E5EA",
    },
    "dark": {
        "bg": "#1C1C1E", "text": "#F2F2F7", "sub": "#8E8E93",
        "card": "#2C2C2E", "border": "#3A3A3C", "hover": "#3A3A3C",
        "accent": "#FF453A", "normal": "#0A84FF", "idle": "#48484A",
        "mute": "#2C2C2E", "blue": "#0A84FF",
        "glass": "#2C2C2E", "glassBorder": "#3A3A3C",
    },
}


# ---------- 拍子圆点（可编辑） ----------
class BeatDot(QFrame):
    STATES = ["normal", "accent", "mute"]

    def __init__(self, beat, on_change):
        super().__init__()
        self.beat = beat
        self.on_change = on_change
        self.active = False
        self.dark = False
        self.setFixedSize(36, 50)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("左键：重拍/普通/静音  ·  右键：细分 1~4")

    def set_active(self, a):
        if self.active != a:
            self.active = a
            self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            i = self.STATES.index(self.beat["state"])
            self.beat["state"] = self.STATES[(i + 1) % len(self.STATES)]
        elif e.button() == Qt.RightButton:
            self.beat["sub"] = self.beat["sub"] % 4 + 1
        self.update()
        self.on_change()

    def paintEvent(self, e):
        th = THEMES["dark" if self.dark else "light"]
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        state = self.beat["state"]
        sub = self.beat["sub"]

        if state == "accent":
            base = QColor(th["accent"])
        elif state == "mute":
            base = QColor(th["mute"])
        else:
            base = QColor(th["normal"])

        cx, top, r = 18, 6, 13
        if self.active and state != "mute":
            p.setBrush(QBrush(base))
            p.setPen(Qt.NoPen)
        else:
            if state == "mute":
                p.setPen(QPen(QColor(th["idle"]), 1.5, Qt.DashLine))
                p.setBrush(Qt.NoBrush)
            else:
                faded = QColor(base)
                faded.setAlpha(90)
                p.setBrush(QBrush(faded))
                p.setPen(Qt.NoPen)
        p.drawEllipse(cx - r // 2 - 3, top, r, r)

        if sub > 1 and state != "mute":
            dot_r, gap, y = 4, 8, 34
            start_x = cx - (sub - 1) * gap // 2
            for i in range(sub):
                c = QColor(base)
                c.setAlpha(220 if i == 0 else 120)
                p.setBrush(QBrush(c))
                p.setPen(Qt.NoPen)
                p.drawEllipse(start_x + i * gap - dot_r // 2, y, dot_r, dot_r)


# ---------- 音频引擎：采样级精确计时（无锁，避免死锁） ----------
class AudioEngine:
    def __init__(self, get_next_event):
        self.get_next_event = get_next_event
        self.stream = None
        self.frames_per_tick = 0
        self.frame_counter = 0
        self.active_voices = []
        self.running = False

    def set_interval(self, seconds):
        self.frames_per_tick = max(1, int(SR * seconds))

    def start(self, interval_sec):
        if not HAS_SOUND:
            return False
        self.set_interval(interval_sec)
        self.frame_counter = 0
        self.active_voices = []
        self.running = True
        try:
            self.stream = sd.OutputStream(
                samplerate=SR, channels=1, dtype="float32",
                blocksize=512, callback=self._callback)
            self.stream.start()
            return True
        except Exception as e:
            print("音频启动失败:", e)
            self.running = False
            return False

    def stop(self):
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def _callback(self, outdata, frames, time_info, status):
        out = np.zeros(frames, dtype=np.float32)
        idx = 0
        remaining = frames
        while remaining > 0:
            if self.frame_counter <= 0:
                wave = self.get_next_event()
                if wave is not None:
                    self.active_voices.append([wave, 0])
                if self.frames_per_tick <= 0:
                    self.frames_per_tick = SR
                self.frame_counter = self.frames_per_tick
            step = min(self.frame_counter, remaining)
            for v in self.active_voices:
                w, pos = v
                n = min(step, len(w) - pos)
                if n > 0:
                    out[idx:idx + n] += w[pos:pos + n]
                    v[1] += n
            self.active_voices = [v for v in self.active_voices
                                  if v[1] < len(v[0])]
            idx += step
            remaining -= step
            self.frame_counter -= step
        np.clip(out, -1.0, 1.0, out=out)
        outdata[:, 0] = out


# ---------- 主窗口 ----------
class Metronome(QWidget):
    SPEED_TERMS = [
        (40, "Largo"), (60, "Larghetto"), (66, "Adagio"),
        (76, "Andante"), (108, "Moderato"), (120, "Allegro"),
        (168, "Presto"), (200, "Prestissimo"), (10000, "")
    ]

    def __init__(self):
        super().__init__()
        self.bpm = 120
        self.beats_per_bar = 4
        self.volume = 0.7
        self.dark = False
        self.is_playing = False
        self.mini_mode = False
        self.tap_times = []
        self._drag_pos = None
        self._current_sound = "Click（清脆）"

        self.beats = [{"state": "accent" if i == 0 else "normal", "sub": 1}
                      for i in range(self.beats_per_bar)]

        self._play_beat = 0
        self._play_sub = 0
        self._visual_beat = 0

        self.engine = AudioEngine(self._next_event)

        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self._refresh_active_dot)

        self.init_ui()
        self.apply_theme()
        self.setFocusPolicy(Qt.StrongFocus)

    # ---------------- UI ----------------
    def init_ui(self):
        self.setWindowTitle("Metronome")
        self.normal_min_w = 400

        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(26, 22, 26, 22)
        self.root.setSpacing(14)

        # 顶部栏
        top = QHBoxLayout()
        self.term_label = QLabel("Allegro")
        self.term_label.setObjectName("term")
        top.addWidget(self.term_label)
        top.addStretch()
        self.theme_btn = QPushButton("🌙")
        self.theme_btn.setObjectName("icon")
        self.theme_btn.setFixedSize(34, 34)
        self.theme_btn.clicked.connect(self.toggle_theme)
        top.addWidget(self.theme_btn)
        self.root.addLayout(top)

        # BPM 大数字
        self.bpm_edit = QLineEdit(str(self.bpm))
        self.bpm_edit.setAlignment(Qt.AlignCenter)
        self.bpm_edit.setValidator(QIntValidator(20, 400))
        self.bpm_edit.setObjectName("bpmBig")
        self.bpm_edit.editingFinished.connect(self.on_bpm_typed)
        self.root.addWidget(self.bpm_edit)

        self.unit_label = QLabel("BPM")
        self.unit_label.setAlignment(Qt.AlignCenter)
        self.unit_label.setObjectName("unit")
        self.root.addWidget(self.unit_label)

        # 微调 + 滑块
        adj = QHBoxLayout()
        self.minus_btn = QPushButton("−")
        self.plus_btn = QPushButton("+")
        for b in (self.minus_btn, self.plus_btn):
            b.setObjectName("round")
            b.setFixedSize(40, 40)
        self.minus_btn.clicked.connect(lambda: self.change_bpm(-1))
        self.plus_btn.clicked.connect(lambda: self.change_bpm(1))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(20, 400)
        self.slider.setValue(self.bpm)
        self.slider.valueChanged.connect(self.on_slider)
        adj.addWidget(self.minus_btn)
        adj.addWidget(self.slider)
        adj.addWidget(self.plus_btn)
        self.root.addLayout(adj)

        # 快速 + Tap
        quick = QHBoxLayout()
        for v in (60, 90, 120, 140, 180):
            qb = QPushButton(str(v))
            qb.setObjectName("chip")
            qb.clicked.connect(lambda _, x=v: self.set_bpm(x))
            quick.addWidget(qb)
        self.tap_btn = QPushButton("TAP")
        self.tap_btn.setObjectName("chip")
        self.tap_btn.clicked.connect(self.tap_tempo)
        quick.addWidget(self.tap_btn)
        self.root.addLayout(quick)

        # 提示
        self.hint = QLabel("圆点：左键 重拍/普通/静音 · 右键 细分　|　空格启停 ↑↓调速")
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setObjectName("hint")
        self.root.addWidget(self.hint)

        # 拍子圆点
        self.dots_layout = QHBoxLayout()
        self.dots_layout.setAlignment(Qt.AlignCenter)
        self.dots_layout.setSpacing(8)
        self.root.addLayout(self.dots_layout)
        self.build_dots()

        # 控制网格
        grid = QGridLayout()
        grid.setVerticalSpacing(10)
        self.beats_label = QLabel("每小节拍数")
        grid.addWidget(self.beats_label, 0, 0)
        self.beats_spin = QSpinBox()
        self.beats_spin.setRange(1, 12)
        self.beats_spin.setValue(self.beats_per_bar)
        self.beats_spin.valueChanged.connect(self.on_beats_changed)
        grid.addWidget(self.beats_spin, 0, 1)

        self.sound_label = QLabel("声音")
        grid.addWidget(self.sound_label, 1, 0)
        self.sound_combo = QComboBox()
        self.sound_combo.addItems(list(SOUND_PRESETS.keys()))
        self.sound_combo.currentTextChanged.connect(
            lambda s: setattr(self, "_current_sound", s))
        grid.addWidget(self.sound_combo, 1, 1)

        self.vol_label = QLabel("音量")
        grid.addWidget(self.vol_label, 2, 0)
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(70)
        self.vol_slider.valueChanged.connect(
            lambda v: setattr(self, "volume", v / 100))
        grid.addWidget(self.vol_slider, 2, 1)
        self.root.addLayout(grid)

        # 选项
        opts = QHBoxLayout()
        self.top_chk = QCheckBox("置顶窗口")
        self.top_chk.stateChanged.connect(self.toggle_on_top)
        self.mini_chk = QCheckBox("迷你模式")
        self.mini_chk.stateChanged.connect(lambda s: self.set_mini(bool(s)))
        opts.addWidget(self.top_chk)
        opts.addWidget(self.mini_chk)
        self.root.addLayout(opts)

        # 启停
        self.play_btn = QPushButton("▶  开始")
        self.play_btn.setObjectName("play")
        self.play_btn.setFixedHeight(54)
        self.play_btn.clicked.connect(self.toggle_play)
        self.root.addWidget(self.play_btn)

        # ---- 迷你条（普通实心，无特效） ----
        self.mini_bar = QWidget(self)
        self.mini_bar.setObjectName("miniBar")
        mb = QHBoxLayout(self.mini_bar)
        mb.setContentsMargins(8, 0, 10, 0)
        mb.setSpacing(10)

        self.mini_play = QPushButton("▶")
        self.mini_play.setObjectName("miniPlay")
        self.mini_play.setFixedSize(32, 32)
        self.mini_play.setCursor(Qt.PointingHandCursor)
        self.mini_play.clicked.connect(self.toggle_play)

        self.mini_dot = QLabel("●")
        self.mini_dot.setObjectName("miniDot")
        self.mini_dot.setFixedWidth(14)
        self.mini_dot.setAlignment(Qt.AlignCenter)

        self.mini_bpm = QLabel(str(self.bpm))
        self.mini_bpm.setObjectName("miniBpm")
        self.mini_bpm.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

        self.mini_unit = QLabel("BPM")
        self.mini_unit.setObjectName("miniUnit")
        self.mini_unit.setAlignment(Qt.AlignBottom | Qt.AlignLeft)

        step_box = QVBoxLayout()
        step_box.setSpacing(1)
        step_box.setContentsMargins(0, 0, 0, 0)
        self.mini_plus = QPushButton("＋")
        self.mini_minus = QPushButton("−")
        for b in (self.mini_plus, self.mini_minus):
            b.setObjectName("miniStep")
            b.setFixedSize(24, 15)
            b.setCursor(Qt.PointingHandCursor)
        self.mini_plus.clicked.connect(lambda: self.change_bpm(1))
        self.mini_minus.clicked.connect(lambda: self.change_bpm(-1))
        step_box.addWidget(self.mini_plus)
        step_box.addWidget(self.mini_minus)

        sep = QFrame()
        sep.setObjectName("miniSep")
        sep.setFixedSize(1, 22)

        self.mini_pin = QPushButton("📌")
        self.mini_pin.setObjectName("miniIcon")
        self.mini_pin.setFixedSize(26, 26)
        self.mini_pin.setCheckable(True)
        self.mini_pin.setCursor(Qt.PointingHandCursor)
        self.mini_pin.setToolTip("置顶")
        self.mini_pin.clicked.connect(self.toggle_mini_pin)

        self.mini_back = QPushButton("⤢")
        self.mini_back.setObjectName("miniIcon")
        self.mini_back.setFixedSize(26, 26)
        self.mini_back.setCursor(Qt.PointingHandCursor)
        self.mini_back.setToolTip("返回正常模式")
        self.mini_back.clicked.connect(lambda: self.set_mini(False))

        mb.addWidget(self.mini_play)
        mb.addWidget(self.mini_dot)
        mb.addWidget(self.mini_bpm)
        mb.addWidget(self.mini_unit)
        mb.addStretch()
        mb.addLayout(step_box)
        mb.addWidget(sep)
        mb.addWidget(self.mini_pin)
        mb.addWidget(self.mini_back)
        self.mini_bar.hide()

        self.update_display()

    def build_dots(self):
        while self.dots_layout.count():
            w = self.dots_layout.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.dots = []
        for i in range(self.beats_per_bar):
            d = BeatDot(self.beats[i], self.on_beats_edited)
            d.dark = self.dark
            self.dots_layout.addWidget(d)
            self.dots.append(d)
    # ---------------- 逻辑 ----------------
    def on_beats_edited(self):
        pass

    def speed_term(self, bpm):
        for limit, name in self.SPEED_TERMS:
            if bpm < limit:
                return name
        return ""

    def update_display(self):
        self.bpm_edit.setText(str(self.bpm))
        self.mini_bpm.setText(str(self.bpm))
        self.term_label.setText(self.speed_term(self.bpm))
        if self.slider.value() != self.bpm:
            self.slider.blockSignals(True)
            self.slider.setValue(self.bpm)
            self.slider.blockSignals(False)

    def set_bpm(self, v):
        self.bpm = max(20, min(400, int(v)))
        self.update_display()
        if self.is_playing:
            self.engine.set_interval(self._sub_interval())

    def change_bpm(self, d):
        self.set_bpm(self.bpm + d)

    def on_bpm_typed(self):
        try:
            self.set_bpm(int(self.bpm_edit.text()))
        except ValueError:
            self.update_display()

    def on_slider(self, v):
        self.set_bpm(v)

    def on_beats_changed(self, v):
        self.beats_per_bar = v
        while len(self.beats) < v:
            self.beats.append({"state": "normal", "sub": 1})
        self.beats = self.beats[:v]
        self.build_dots()
        self.apply_theme()

    def tap_tempo(self):
        now = time.time()
        self.tap_times.append(now)
        self.tap_times = [t for t in self.tap_times if now - t < 2.0]
        if len(self.tap_times) >= 2:
            bpm = 60.0 / np.mean(np.diff(self.tap_times))
            self.set_bpm(round(bpm))

    # ---------------- 播放 ----------------
    def _sub_interval(self):
        sub = self.beats[self._play_beat]["sub"]
        return 60.0 / self.bpm / sub

    def _next_event(self):
        beat = self.beats[self._play_beat]
        state = beat["state"]
        sub = beat["sub"]
        is_main = (self._play_sub == 0)

        wave = None
        if state != "mute":
            cache = SOUND_CACHE[self._current_sound]
            if state == "accent" and is_main:
                wave = cache["accent"]
            elif is_main:
                wave = cache["normal"]
            else:
                wave = cache["sub"]
            wave = wave * self.volume

        if is_main:
            self._visual_beat = self._play_beat

        self._play_sub += 1
        if self._play_sub >= sub:
            self._play_sub = 0
            self._play_beat = (self._play_beat + 1) % self.beats_per_bar

        self.engine.frames_per_tick = max(1, int(SR * self._sub_interval()))
        return wave

    def toggle_play(self):
        self.stop() if self.is_playing else self.start()

    def start(self):
        self.is_playing = True
        self._play_beat = 0
        self._play_sub = 0
        self._visual_beat = 0
        self.play_btn.setText("■  停止")
        self.mini_play.setText("■")
        if not self.engine.start(self._sub_interval()):
            self._fallback_timer()
        self.ui_timer.start(30)

    def stop(self):
        self.is_playing = False
        self.engine.stop()
        self.ui_timer.stop()
        if hasattr(self, "_fb_timer"):
            self._fb_timer.stop()
        self.play_btn.setText("▶  开始")
        self.mini_play.setText("▶")
        for d in self.dots:
            d.set_active(False)
        self.mini_dot.setText("●")

    def _fallback_timer(self):
        self._fb_timer = QTimer()
        self._fb_timer.timeout.connect(self._next_event)
        self._fb_timer.start(int(self._sub_interval() * 1000))

    def _refresh_active_dot(self):
        for i, d in enumerate(self.dots):
            d.set_active(i == self._visual_beat)
        th = THEMES["dark" if self.dark else "light"]
        state = self.beats[self._visual_beat]["state"]
        color = th["accent"] if state == "accent" else th["normal"]
        self.mini_dot.setStyleSheet(f"color:{color};")

    # ---------------- 键盘快捷键 ----------------
    def keyPressEvent(self, e):
        key = e.key()
        editing = self.bpm_edit.hasFocus()
        if key == Qt.Key_Space:
            self.toggle_play()
            e.accept(); return
        if key == Qt.Key_Up:
            self.change_bpm(5 if e.modifiers() & Qt.ShiftModifier else 1)
            e.accept(); return
        if key == Qt.Key_Down:
            self.change_bpm(-5 if e.modifiers() & Qt.ShiftModifier else -1)
            e.accept(); return
        if key == Qt.Key_M:
            self.set_mini(not self.mini_mode)
            e.accept(); return
        if not editing:
            super().keyPressEvent(e)

    # ---------------- 迷你模式（无边框 + 不透明，求稳） ----------------
    def set_mini(self, on):
        self.mini_mode = on
        self.mini_chk.blockSignals(True)
        self.mini_chk.setChecked(on)
        self.mini_chk.blockSignals(False)

        bar_w, bar_h = 260, 56

        if on:
            self._set_main_visible(False)
            self.root.setContentsMargins(0, 0, 0, 0)
            self.setMinimumSize(0, 0)
            self.setMaximumSize(16777215, 16777215)
            self.clearMask()
            self.setAttribute(Qt.WA_TranslucentBackground, False)  # 不透明，求稳
            # 无边框 + 置顶（不带 Tool、不透明 → 不触发 layered window 崩溃）
            self.setWindowFlags(
                Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self.mini_pin.setChecked(True)
            self.mini_pin.setText("📌")
            self.mini_bar.show()
            self.mini_bar.raise_()
            self.show()
            QTimer.singleShot(0, lambda: self._fit_mini(bar_w, bar_h))
        else:
            self.clearMask()
            self.mini_bar.hide()
            self.setAttribute(Qt.WA_TranslucentBackground, False)
            self._set_main_visible(True)
            self.root.setContentsMargins(26, 22, 26, 22)
            self.setWindowFlags(Qt.Window)
            self.setWindowFlag(Qt.WindowStaysOnTopHint,
                               self.top_chk.isChecked())
            self.setMinimumSize(self.normal_min_w, 0)
            self.setMaximumSize(16777215, 16777215)
            self.show()
            QTimer.singleShot(0, lambda: self.resize(
                self.normal_min_w, self.sizeHint().height()))
        self.apply_theme()

    def _fit_mini(self, bar_w, bar_h):
        self.setFixedSize(bar_w, bar_h)
        self.mini_bar.setGeometry(0, 0, bar_w, bar_h)
        self.mini_bar.show()
        self.mini_bar.raise_()

    def _set_main_visible(self, vis):
        def walk(layout):
            for i in range(layout.count()):
                it = layout.itemAt(i)
                if it.widget():
                    it.widget().setVisible(vis)
                elif it.layout():
                    walk(it.layout())
        walk(self.root)

    def toggle_mini_pin(self):
        on = self.mini_pin.isChecked()
        self.mini_pin.setText("📌" if on else "📍")
        self.setWindowFlag(Qt.WindowStaysOnTopHint, on)
        self.show()

    # 迷你条可拖动（无边框靠鼠标拖动）
    def mousePressEvent(self, e):
        if self.mini_mode and e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self.mini_mode and self._drag_pos and e.buttons() & Qt.LeftButton:
            self.move(e.globalPos() - self._drag_pos)

    def mouseReleaseEvent(self, e):
        self._drag_pos = None

    # ---------------- 主题 / 窗口 ----------------
    def toggle_theme(self):
        self.dark = not self.dark
        self.theme_btn.setText("☀️" if self.dark else "🌙")
        for d in self.dots:
            d.dark = self.dark
            d.update()
        self.apply_theme()

    def toggle_on_top(self, state):
        if not self.mini_mode:
            self.setWindowFlag(Qt.WindowStaysOnTopHint, bool(state))
            self.show()

    def apply_theme(self):
        th = THEMES["dark" if self.dark else "light"]
        self.setStyleSheet(f"""
            QWidget {{
                background: {th['bg']}; color: {th['text']};
                font-family: 'Microsoft YaHei', '微软雅黑', sans-serif;
            }}
            #term {{ font-size: 14px; color: {th['sub']}; letter-spacing: 1px; }}
            #bpmBig {{
                font-size: 70px; font-weight: 600;
                border: none; background: transparent; color: {th['text']};
            }}
            #unit {{ font-size: 12px; color: {th['sub']}; letter-spacing: 3px; }}
            #hint {{ font-size: 11px; color: {th['sub']}; }}
            QLabel {{ font-size: 14px; color: {th['text']}; }}

            #icon {{
                background: {th['card']}; border: 1px solid {th['border']};
                border-radius: 17px; font-size: 15px;
            }}
            #icon:hover {{ background: {th['hover']}; }}

            #round {{
                background: {th['card']}; border: 1px solid {th['border']};
                border-radius: 20px; font-size: 22px; color: {th['blue']};
            }}
            #round:hover {{ background: {th['hover']}; }}

            #chip {{
                background: {th['card']}; border: 1px solid {th['border']};
                border-radius: 14px; padding: 8px 10px;
                font-size: 13px; color: {th['blue']}; font-weight: 500;
            }}
            #chip:hover {{ background: {th['blue']}; color: white; }}

            #play {{
                background: {th['blue']}; color: white; border: none;
                border-radius: 16px; font-size: 18px; font-weight: 600;
            }}
            #play:hover {{ background: {th['normal']}; }}

            QComboBox, QSpinBox {{
                background: {th['card']}; border: 1px solid {th['border']};
                border-radius: 10px; padding: 6px 10px; min-height: 24px;
                color: {th['text']};
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background: {th['card']}; border: 1px solid {th['border']};
                border-radius: 10px; selection-background-color: {th['blue']};
                selection-color: white; outline: none; color: {th['text']};
            }}

            QSlider::groove:horizontal {{
                height: 4px; background: {th['border']}; border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: {th['blue']}; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {th['card']}; border: 1px solid {th['border']};
                width: 20px; height: 20px; margin: -8px 0; border-radius: 10px;
            }}

            QCheckBox {{ font-size: 13px; color: {th['text']}; spacing: 6px; }}
            QCheckBox::indicator {{
                width: 18px; height: 18px; border-radius: 5px;
                border: 1px solid {th['border']}; background: {th['card']};
            }}
            QCheckBox::indicator:checked {{
                background: {th['blue']}; border: 1px solid {th['blue']};
            }}

            /* ---- 迷你条（实心，无特效） ---- */
            #miniBar {{
                background: {th['glass']};
                border: 1px solid {th['glassBorder']};
                border-radius: 10px;
            }}
            #miniPlay {{
                background: {th['blue']}; color: white; border: none;
                border-radius: 16px; font-size: 13px; font-weight: 600;
                padding-left: 2px;
            }}
            #miniPlay:hover {{ background: {th['normal']}; }}
            #miniDot {{ font-size: 11px; color: {th['normal']}; }}
            #miniBpm {{
                font-size: 22px; font-weight: 700; color: {th['text']};
                min-width: 42px;
            }}
            #miniUnit {{
                font-size: 9px; color: {th['sub']};
                letter-spacing: 1px; padding-bottom: 4px;
            }}
            #miniStep {{
                background: transparent; color: {th['sub']};
                border: 1px solid {th['border']}; border-radius: 5px;
                font-size: 11px; font-weight: 600; padding: 0;
            }}
            #miniStep:hover {{
                background: {th['blue']}; color: white; border-color: {th['blue']};
            }}
            #miniSep {{ background: {th['border']}; border: none; }}
            #miniIcon {{
                background: transparent; border: none;
                border-radius: 13px; font-size: 13px; color: {th['sub']};
            }}
            #miniIcon:hover {{ background: {th['hover']}; }}
            #miniIcon:checked {{ background: {th['blue']}; }}
        """)

    def closeEvent(self, e):
        self.engine.stop()
        e.accept()


if __name__ == "__main__":
    # 必须在 QApplication 创建之前设置
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))   # 全局微软雅黑
    w = Metronome()
    w.show()
    sys.exit(app.exec_())
