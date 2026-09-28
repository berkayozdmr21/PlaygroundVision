"""
desktop_app.py
================
Çocuk Oyun Alanı — Masaüstü İzleme Uygulaması (TEST)

PySide6 (Qt) arayüzü. Tüm görüntü işleme mantığı `tracking_core.SafetyMonitor`
içinde. SADECE PC kamerasıyla çalışır — mobil/backend kısmı yok.

KURULUM:  pip install -r requirements.txt
ÇALIŞTIRMA:  python desktop_app.py
"""

import sys
import time
import collections

import cv2

from PySide6.QtCore import (
    Qt, QThread, Signal, QTimer, QPropertyAnimation, QEasingCurve,
    QRectF, Property,
)
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QListWidget, QListWidgetItem, QFrame, QSlider,
    QCheckBox, QGraphicsDropShadowEffect, QSizePolicy,
)

from tracking_core import (
    SafetyMonitor, KP_MIN_CONF,
    DEFAULT_MISSING_ALARM_SECONDS, DEFAULT_GROUND_ALARM_SECONDS,
)
from clip_recorder import ClipRecorder

TARGET_WIDTH = 960
TARGET_HEIGHT = 540

SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# --- Palet (BGR — OpenCV çizimleri için) --- #
C_NORMAL = (120, 200, 100)     # yeşil
C_TARGET = (70, 120, 255)      # turuncu-kırmızı
C_FIGHT = (210, 70, 190)       # mor
C_SKELETON = (235, 200, 90)    # camgöbeği

# --- Palet (arayüz) --- #
BG_DEEP = "#0a0c10"
BG = "#0e1116"
SURFACE = "#161b24"
SURFACE_2 = "#1c222d"
BORDER = "#252c39"
TEXT = "#e8eaee"
TEXT_DIM = "#8a929f"
TEXT_FAINT = "#5a616d"
ACCENT = "#3b82f6"
ACCENT_HI = "#2563eb"
OK = "#22c55e"
WARN = "#f59e0b"
DANGER = "#ef4444"
FIGHT = "#a855f7"


# =========================================================================== #
#  KAMERA / İŞLEME THREAD'İ
# =========================================================================== #

class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    status_changed = Signal(dict)
    event_logged = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, camera_index=0,
                 missing_alarm_seconds=DEFAULT_MISSING_ALARM_SECONDS,
                 ground_alarm_seconds=DEFAULT_GROUND_ALARM_SECONDS,
                 fight_enabled=True, use_violence_ai=False, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self._running = False
        self._pending_click = None
        self._pending_clip = None
        self._pending_ai_toggle = None
        self._auto_clip_on_fight = True
        self._fight_was_alarm = False
        self._frame_times = collections.deque(maxlen=30)

        self.monitor = SafetyMonitor(
            missing_alarm_seconds=missing_alarm_seconds,
            ground_alarm_seconds=ground_alarm_seconds,
            fight_enabled=fight_enabled,
            use_violence_ai=use_violence_ai,
        )
        self.monitor.on_event = self.event_logged.emit
        self.recorder = ClipRecorder(fps=15.0)
        self.recorder.on_event = self.event_logged.emit

    # --- Ana thread'den komutlar --- #
    def request_select(self, x, y):
        self._pending_click = (x, y)

    def request_reset(self):
        self.monitor.reset_target()

    def request_clip(self, label):
        self._pending_clip = (label,)

    def set_auto_clip_on_fight(self, enabled):
        self._auto_clip_on_fight = bool(enabled)

    def set_missing_alarm_seconds(self, seconds):
        self.monitor.target.missing_alarm_seconds = seconds

    def set_ground_alarm_seconds(self, seconds):
        self.monitor.fall.ground_seconds = seconds

    def set_fight_enabled(self, enabled):
        self.monitor.set_fight_enabled(enabled)

    def set_fight_sensitivity(self, threshold):
        self.monitor.set_fight_sensitivity(threshold)

    def set_violence_ai_enabled(self, enabled):
        self._pending_ai_toggle = bool(enabled)

    def stop(self):
        self._running = False

    # --- Döngü --- #
    def run(self):
        self._running = True
        cap = None
        try:
            self.event_logged.emit("Model yükleniyor (ilk seferde biraz sürebilir)...")
            self.monitor.load_model()

            self.event_logged.emit("Kamera açılıyor...")
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                self.error_occurred.emit("Kamera açılamadı. Başka bir uygulama kullanıyor olabilir.")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
            self.event_logged.emit("Kamera hazır. Bir kutuya tıklayarak hedef seç.")

            while self._running:
                ok, frame = cap.read()
                if not ok:
                    self.error_occurred.emit("Kameradan kare okunamadı.")
                    break
                if frame.shape[1] != TARGET_WIDTH or frame.shape[0] != TARGET_HEIGHT:
                    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

                if self._pending_click is not None:
                    x, y = self._pending_click
                    self._pending_click = None
                    if not self.monitor.select_target_at(x, y):
                        self.event_logged.emit("Tıklanan noktada kimse tespit edilmedi.")

                if self._pending_ai_toggle is not None:
                    want = self._pending_ai_toggle
                    self._pending_ai_toggle = None
                    self.monitor.set_violence_ai_enabled(want)

                self.recorder.add(frame)
                status = self.monitor.process(frame)

                if self._pending_clip is not None:
                    label = self._pending_clip[0]
                    self._pending_clip = None
                    self.recorder.save(label, note=f"manuel | {time.strftime('%Y-%m-%d %H:%M:%S')}")

                fight_alarm = status["alarm_active"] and status["alarm_reason"] == "fight"
                if self._auto_clip_on_fight and fight_alarm and not self._fight_was_alarm:
                    self.recorder.save("kavga_otomatik",
                                       note=f"otomatik | AI={status.get('fight_last_ai_score')}")
                self._fight_was_alarm = fight_alarm

                self._frame_times.append(time.time())
                status["fps"] = self._fps()

                self._draw_overlay(frame, status)
                self.frame_ready.emit(self._to_qimage(frame))
                self.status_changed.emit(status)

        except Exception as exc:
            self.error_occurred.emit(f"Beklenmeyen hata: {exc}")
        finally:
            self.recorder.close()
            self.monitor.shutdown()
            if cap is not None:
                cap.release()
            self.event_logged.emit("Kamera kapatıldı.")

    def _fps(self):
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    # --- Çizim --- #
    def _draw_overlay(self, frame, status):
        target_id = status["target_id"]
        tracks = self.monitor.tracker.tracks
        fighters = set()
        for a, b in status.get("fighting_pairs", []):
            fighters.add(a)
            fighters.add(b)

        for track_id, track in tracks.items():
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            is_target = (track_id == target_id)
            is_fight = track_id in fighters
            color = C_FIGHT if is_fight else (C_TARGET if is_target else C_NORMAL)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"HEDEF {track_id}" if is_target else str(track_id)
            if is_target and status["fall_state"] == "monitoring":
                label += f" · YERDE {int(status['seconds_on_ground'])}s"
            elif is_target and status["fall_state"] == "alarm":
                label += " · DUSME"
            if is_fight:
                label += " · KAVGA?"
            self._chip(frame, label, x1, y1, color)

            if is_target and track.keypoints is not None:
                self._draw_skeleton(frame, track.keypoints)

        for a, b in status.get("fighting_pairs", []):
            if a in tracks and b in tracks:
                ca = self._center(tracks[a].bbox)
                cb = self._center(tracks[b].bbox)
                cv2.line(frame, ca, cb, C_FIGHT, 2)

        if status["alarm_active"]:
            reason = status.get("alarm_reason")
            bc = C_FIGHT if reason == "fight" else (0, 0, 255)
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), bc, 6)

    @staticmethod
    def _center(bbox):
        return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

    @staticmethod
    def _chip(frame, text, x, y, color):
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        y = max(th + 8, y)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 10, y), color, -1)
        cv2.putText(frame, text, (x + 5, y - 5), font, scale, (20, 20, 20), thick, cv2.LINE_AA)

    @staticmethod
    def _draw_skeleton(frame, keypoints):
        for a, b in SKELETON_EDGES:
            if a < len(keypoints) and b < len(keypoints):
                xa, ya, ca = keypoints[a]
                xb, yb, cb = keypoints[b]
                if ca >= KP_MIN_CONF and cb >= KP_MIN_CONF:
                    cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), C_SKELETON, 2, cv2.LINE_AA)
        for x, y, c in keypoints:
            if c >= KP_MIN_CONF:
                cv2.circle(frame, (int(x), int(y)), 3, C_SKELETON, -1, cv2.LINE_AA)

    @staticmethod
    def _to_qimage(frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


# =========================================================================== #
#  ÖZEL WIDGET'LAR
# =========================================================================== #

class ToggleSwitch(QCheckBox):
    """iOS tarzı, animasyonlu aç/kapa anahtarı."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pos = 0.0
        self.setFixedSize(46, 26)
        self.setCursor(Qt.PointingHandCursor)
        self._anim = QPropertyAnimation(self, b"knob", self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QEasingCurve.InOutCubic)
        self.toggled.connect(self._go)

    def _go(self, checked):
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def get_knob(self):
        return self._pos

    def set_knob(self, v):
        self._pos = v
        self.update()

    knob = Property(float, get_knob, set_knob)

    def hitButton(self, pos):
        return self.rect().contains(pos)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        on = QColor(ACCENT)
        off = QColor("#39404d")
        track = QColor(
            int(off.red() + (on.red() - off.red()) * self._pos),
            int(off.green() + (on.green() - off.green()) * self._pos),
            int(off.blue() + (on.blue() - off.blue()) * self._pos),
        )
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0, 0, self.width(), self.height()), 13, 13)
        d = 20.0
        x = 3 + (self.width() - d - 6) * self._pos
        p.setBrush(QColor("#ffffff"))
        p.drawEllipse(QRectF(x, 3, d, d))


class PulseDot(QWidget):
    """Yanıp sönen küçük durum noktası."""

    def __init__(self, color=OK, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._a = 1.0
        self.setFixedSize(14, 14)
        self._anim = QPropertyAnimation(self, b"alpha", self)
        self._anim.setDuration(900)
        self._anim.setStartValue(1.0)
        self._anim.setKeyValueAt(0.5, 0.25)
        self._anim.setEndValue(1.0)
        self._anim.setLoopCount(-1)
        self._anim.start()

    def set_color(self, color):
        self._color = QColor(color)
        self.update()

    def get_alpha(self):
        return self._a

    def set_alpha(self, v):
        self._a = v
        self.update()

    alpha = Property(float, get_alpha, set_alpha)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QColor(self._color)
        c.setAlphaF(0.18)
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        p.drawEllipse(self.rect())
        c2 = QColor(self._color)
        c2.setAlphaF(self._a)
        p.setBrush(c2)
        p.drawEllipse(self.rect().adjusted(3, 3, -3, -3))


class Card(QFrame):
    """Başlıklı, gölgeli konteyner."""

    def __init__(self, title=None, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._v = QVBoxLayout(self)
        self._v.setContentsMargins(16, 14, 16, 16)
        self._v.setSpacing(12)
        if title:
            lbl = QLabel(title)
            lbl.setObjectName("cardTitle")
            self._v.addWidget(lbl)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setXOffset(0)
        shadow.setYOffset(6)
        shadow.setColor(QColor(0, 0, 0, 110))
        self.setGraphicsEffect(shadow)

    def add(self, w):
        self._v.addWidget(w)

    def add_layout(self, lay):
        self._v.addLayout(lay)


class StatTile(QFrame):
    """Büyük değer + etiket + renkli nokta."""

    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.setObjectName("tile")
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(2)
        top = QHBoxLayout()
        top.setSpacing(7)
        self.dot = QLabel("●")
        self.dot.setObjectName("tileDot")
        self.value = QLabel("—")
        self.value.setObjectName("tileValue")
        top.addWidget(self.dot)
        top.addWidget(self.value)
        top.addStretch()
        cap = QLabel(label)
        cap.setObjectName("tileLabel")
        v.addLayout(top)
        v.addWidget(cap)

    def set(self, value, color=TEXT_DIM):
        self.value.setText(str(value))
        self.dot.setStyleSheet(f"#tileDot {{ color: {color}; font-size: 11px; }}")
        self.value.setStyleSheet(f"#tileValue {{ color: {TEXT}; font-size: 20px; font-weight: 700; }}")


class MonitorRow(QWidget):
    """İzleme satırı: ad + (opsiyonel) kaydırıcı + aç/kapa anahtarı."""

    def __init__(self, name, has_slider=False, slider_range=(3, 30),
                 slider_default=10, unit="sn", parent=None):
        super().__init__(parent)
        self.unit = unit
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 4, 0, 4)
        v.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.name = QLabel(name)
        self.name.setObjectName("rowName")
        self.toggle = ToggleSwitch()
        row.addWidget(self.name)
        row.addStretch()
        if has_slider:
            self.value_lbl = QLabel(f"{slider_default} {unit}")
            self.value_lbl.setObjectName("rowValue")
            row.addWidget(self.value_lbl)
        row.addWidget(self.toggle)
        v.addLayout(row)

        self.slider = None
        if has_slider:
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(*slider_range)
            self.slider.setValue(slider_default)
            self.slider.valueChanged.connect(
                lambda x: self.value_lbl.setText(f"{x} {self.unit}"))
            v.addWidget(self.slider)


# =========================================================================== #
#  ANA PENCERE
# =========================================================================== #

class VideoPanel(QFrame):
    clicked = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setObjectName("videoPanel")
        self.setFixedSize(TARGET_WIDTH + 4, TARGET_HEIGHT + 40)
        self._pix = None

        self.image = QLabel(self)
        self.image.setObjectName("videoImage")
        self.image.setGeometry(2, 34, TARGET_WIDTH, TARGET_HEIGHT)
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setText("Kamera başlatılıyor…")
        self.image.setCursor(Qt.PointingHandCursor)
        self.image.mousePressEvent = self._img_click

        self.live = QLabel("● CANLI", self)
        self.live.setObjectName("liveBadge")
        self.live.move(14, 7)
        self.live.adjustSize()

        self.meta = QLabel("", self)
        self.meta.setObjectName("videoMeta")
        self.meta.move(TARGET_WIDTH - 220, 9)
        self.meta.resize(210, 20)
        self.meta.setAlignment(Qt.AlignRight)

        self.banner = QLabel("", self)
        self.banner.setObjectName("videoBanner")
        self.banner.setGeometry(2, 34, TARGET_WIDTH, 40)
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.hide()

    def _img_click(self, ev):
        if ev.button() == Qt.LeftButton:
            pos = ev.position() if hasattr(ev, "position") else ev.pos()
            self.clicked.emit(int(pos.x()), int(pos.y()))

    def set_frame(self, qimg):
        self.image.setPixmap(QPixmap.fromImage(qimg))

    def set_meta(self, text):
        self.meta.setText(text)

    def set_banner(self, text, color):
        if not text:
            self.banner.hide()
            return
        self.banner.setText(text)
        self.banner.setStyleSheet(
            f"#videoBanner {{ background-color: {color}; color: white; "
            f"font-size: 14px; font-weight: 700; letter-spacing: 1px; }}")
        self.banner.show()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oyun Alanı İzleme")
        self.worker = None
        self._build_ui()
        self._apply_style()
        QTimer.singleShot(100, self._start_worker)

    # ---- UI ---- #
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_header())

        body = QWidget()
        body.setObjectName("body")
        root = QHBoxLayout(body)
        root.setContentsMargins(20, 18, 20, 20)
        root.setSpacing(18)

        # Sol — video
        left = QVBoxLayout()
        left.setSpacing(12)
        self.video = VideoPanel()
        self.video.clicked.connect(self._on_video_clicked)
        left.addWidget(self.video, 0, Qt.AlignHCenter)
        hint = QLabel("Video üstünde bir kişiye tıklayarak HEDEF seç · sonra kadraj dışına çık ya da yere uzan")
        hint.setObjectName("hint")
        hint.setAlignment(Qt.AlignHCenter)
        left.addWidget(hint)
        left.addStretch()
        root.addLayout(left)

        # Sağ — panel
        side = QVBoxLayout()
        side.setSpacing(16)
        side.addWidget(self._build_stats())
        side.addWidget(self._build_monitors())
        side.addWidget(self._build_dataset())
        side.addWidget(self._build_log(), 1)
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(360)
        root.addWidget(side_w)

        outer.addWidget(body, 1)

    def _build_header(self):
        h = QFrame()
        h.setObjectName("header")
        h.setFixedHeight(58)
        lay = QHBoxLayout(h)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(12)

        mark = QLabel("◉")
        mark.setObjectName("brandMark")
        title = QLabel("Oyun Alanı İzleme")
        title.setObjectName("brandTitle")
        sub = QLabel("TEST")
        sub.setObjectName("brandTag")
        lay.addWidget(mark)
        lay.addWidget(title)
        lay.addWidget(sub)
        lay.addStretch()

        self.hdr_state = QLabel("Bağlanıyor…")
        self.hdr_state.setObjectName("hdrState")
        self.hdr_dot = PulseDot(WARN)
        self.hdr_fps = QLabel("")
        self.hdr_fps.setObjectName("hdrFps")
        lay.addWidget(self.hdr_fps)
        lay.addSpacing(6)
        lay.addWidget(self.hdr_dot)
        lay.addWidget(self.hdr_state)
        return h

    def _build_stats(self):
        card = Card()
        grid = QGridLayout()
        grid.setSpacing(10)
        self.tile_people = StatTile("KADRAJDA")
        self.tile_target = StatTile("HEDEF")
        self.tile_alarm = StatTile("ALARM")
        for i, t in enumerate((self.tile_people, self.tile_target, self.tile_alarm)):
            grid.addWidget(t, 0, i)
        card.add_layout(grid)
        return card

    def _build_monitors(self):
        card = Card("İZLEME")
        self.row_missing = MonitorRow("Kadraj dışına çıkma", has_slider=True,
                                      slider_range=(3, 30), slider_default=DEFAULT_MISSING_ALARM_SECONDS)
        self.row_fall = MonitorRow("Düşme · yerde kalma", has_slider=True,
                                   slider_range=(10, 120), slider_default=DEFAULT_GROUND_ALARM_SECONDS)
        self.row_fight = MonitorRow("Kavga · itiş-kakış", has_slider=True,
                                    slider_range=(1, 10), slider_default=5, unit="hassasiyet")
        self.row_ai = MonitorRow("AI kavga doğrulaması")

        for r in (self.row_missing, self.row_fall, self.row_fight):
            r.toggle.setChecked(True)
        self.row_missing.toggle.setEnabled(False)   # kadraj-dışı hep açık
        self.row_fall.toggle.setEnabled(False)      # düşme hep açık

        self.row_missing.slider.valueChanged.connect(
            lambda v: self.worker and self.worker.set_missing_alarm_seconds(v))
        self.row_fall.slider.valueChanged.connect(
            lambda v: self.worker and self.worker.set_ground_alarm_seconds(v))
        self.row_fight.slider.valueChanged.connect(self._on_fight_sensitivity)
        self.row_fight.toggle.toggled.connect(
            lambda c: self.worker and self.worker.set_fight_enabled(c))
        self.row_ai.toggle.toggled.connect(self._on_ai_toggled)

        for r in (self.row_missing, self.row_fall, self.row_fight, self.row_ai):
            card.add(r)
            line = QFrame()
            line.setObjectName("sep")
            line.setFixedHeight(1)
            card.add(line)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.btn_reset = QPushButton("Hedefi sıfırla")
        self.btn_reset.setObjectName("ghost")
        self.btn_reset.clicked.connect(lambda: self.worker and self.worker.request_reset())
        self.btn_cam = QPushButton("Kamerayı durdur")
        self.btn_cam.clicked.connect(self._on_toggle_camera)
        row.addWidget(self.btn_reset)
        row.addWidget(self.btn_cam)
        card.add_layout(row)
        return card

    def _build_dataset(self):
        card = Card("EĞİTİM VERİSİ")
        note = QLabel("Son ~8 sn + sonraki ~3 sn kaydedilir → dataset/<etiket>/")
        note.setObjectName("hint")
        note.setWordWrap(True)
        card.add(note)
        row = QHBoxLayout()
        row.setSpacing(10)
        self.btn_clip_fight = QPushButton("⬤  Kavga klibi")
        self.btn_clip_fight.setObjectName("danger")
        self.btn_clip_fight.clicked.connect(lambda: self._clip("kavga"))
        self.btn_clip_normal = QPushButton("○  Normal klip")
        self.btn_clip_normal.setObjectName("ghost")
        self.btn_clip_normal.clicked.connect(lambda: self._clip("normal"))
        row.addWidget(self.btn_clip_fight)
        row.addWidget(self.btn_clip_normal)
        card.add_layout(row)

        self.row_autoclip = MonitorRow("Kavga alarmında otomatik kaydet")
        self.row_autoclip.toggle.setChecked(True)
        self.row_autoclip.toggle.toggled.connect(
            lambda c: self.worker and self.worker.set_auto_clip_on_fight(c))
        card.add(self.row_autoclip)
        return card

    def _build_log(self):
        card = Card("OLAY GÜNLÜĞÜ")
        self.log = QListWidget()
        self.log.setObjectName("log")
        card.add(self.log)
        return card

    # ---- Stil ---- #
    def _apply_style(self):
        self.setStyleSheet(f"""
            QWidget {{ color: {TEXT}; font-family: 'Segoe UI','Inter',sans-serif; font-size: 13px; }}
            QMainWindow, #body {{ background-color: {BG}; }}
            #header {{ background-color: {BG_DEEP}; border-bottom: 1px solid {BORDER}; }}
            #brandMark {{ color: {ACCENT}; font-size: 18px; }}
            #brandTitle {{ font-size: 15px; font-weight: 700; letter-spacing: .3px; }}
            #brandTag {{ color: {TEXT_FAINT}; font-size: 10px; font-weight: 700;
                         border: 1px solid {BORDER}; border-radius: 5px; padding: 2px 6px; }}
            #hdrState {{ color: {TEXT_DIM}; font-size: 12px; }}
            #hdrFps {{ color: {TEXT_FAINT}; font-size: 11px; }}

            #card {{ background-color: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; }}
            #cardTitle {{ color: {TEXT_FAINT}; font-size: 10px; font-weight: 800; letter-spacing: 1.4px; }}
            #hint {{ color: {TEXT_FAINT}; font-size: 11px; }}
            #sep {{ background-color: {BORDER}; border: none; }}

            #tile {{ background-color: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 10px; }}
            #tileValue {{ font-size: 20px; font-weight: 700; }}
            #tileLabel {{ color: {TEXT_FAINT}; font-size: 9px; font-weight: 800; letter-spacing: 1.2px; }}
            #tileDot {{ color: {TEXT_FAINT}; font-size: 11px; }}

            #rowName {{ font-size: 13px; font-weight: 600; }}
            #rowValue {{ color: {ACCENT}; font-size: 12px; font-weight: 700; }}

            #videoPanel {{ background-color: #000; border: 1px solid {BORDER}; border-radius: 14px; }}
            #videoImage {{ background-color: #06070a; color: {TEXT_FAINT};
                           border-bottom-left-radius: 12px; border-bottom-right-radius: 12px; }}
            #liveBadge {{ color: {DANGER}; font-size: 11px; font-weight: 800; letter-spacing: 1.5px; }}
            #videoMeta {{ color: {TEXT_DIM}; font-size: 11px; }}

            QPushButton {{ background-color: {ACCENT}; color: #fff; border: none;
                           border-radius: 9px; padding: 10px 14px; font-size: 12px; font-weight: 700; }}
            QPushButton:hover {{ background-color: {ACCENT_HI}; }}
            QPushButton#ghost {{ background-color: transparent; border: 1px solid {BORDER}; color: {TEXT_DIM}; }}
            QPushButton#ghost:hover {{ border-color: {ACCENT}; color: {TEXT}; }}
            QPushButton#danger {{ background-color: {DANGER}; }}
            QPushButton#danger:hover {{ background-color: #dc2626; }}

            QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
            QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: #fff; width: 14px; height: 14px;
                                          margin: -6px 0; border-radius: 7px; }}

            #log {{ background-color: {BG_DEEP}; border: 1px solid {BORDER}; border-radius: 10px;
                    font-family: 'Consolas','Menlo',monospace; font-size: 11px; outline: none; }}
            #log::item {{ padding: 3px 4px; border-bottom: 1px solid rgba(255,255,255,0.03); }}
            QScrollBar:vertical {{ background: transparent; width: 8px; margin: 4px; }}
            QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 24px; }}
            QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
        """)

    # ---- Worker ---- #
    def _sensitivity_to_threshold(self, value):
        return 2.9 - 0.2 * value

    def _start_worker(self):
        self.worker = CameraWorker(
            camera_index=0,
            missing_alarm_seconds=self.row_missing.slider.value(),
            ground_alarm_seconds=self.row_fall.slider.value(),
            fight_enabled=self.row_fight.toggle.isChecked(),
            use_violence_ai=self.row_ai.toggle.isChecked(),
        )
        self.worker.frame_ready.connect(self.video.set_frame)
        self.worker.status_changed.connect(self._on_status)
        self.worker.event_logged.connect(self._log_line)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.start()
        self.worker.set_fight_sensitivity(self._sensitivity_to_threshold(self.row_fight.slider.value()))
        self.worker.set_auto_clip_on_fight(self.row_autoclip.toggle.isChecked())
        self.btn_cam.setText("Kamerayı durdur")

    def _stop_worker(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None
        self.btn_cam.setText("Kamerayı başlat")
        self.hdr_state.setText("Kamera kapalı")
        self.hdr_dot.set_color(TEXT_FAINT)
        self.video.set_banner("", "")
        self.video.live.hide()

    # ---- Slotlar ---- #
    def _on_toggle_camera(self):
        if self.worker is not None:
            self._stop_worker()
        else:
            self.video.live.show()
            self._start_worker()

    def _on_video_clicked(self, x, y):
        if self.worker is not None:
            self.worker.request_select(x, y)

    def _on_fight_sensitivity(self, value):
        if self.worker is not None:
            self.worker.set_fight_sensitivity(self._sensitivity_to_threshold(value))

    def _on_ai_toggled(self, checked):
        if self.worker is not None:
            if checked:
                self._log_line("AI modeli yükleniyor, birkaç saniye sürebilir…")
            self.worker.set_violence_ai_enabled(checked)

    def _clip(self, label):
        if self.worker is not None:
            self.worker.request_clip(label)

    def _on_status(self, s):
        # başlık
        self.hdr_fps.setText(f"{s.get('fps', 0):.0f} FPS" if s.get("fps") else "")
        headline, color = self._headline(s)
        self.hdr_state.setText(headline)
        self.hdr_dot.set_color(color)
        self.video.set_meta(
            f"{s['person_count']} kişi"
            + (f"   ·   AI şiddet {s['fight_last_ai_score']:.2f}"
               if s.get("fight_ai_active") and s.get("fight_last_ai_score") is not None else "")
        )

        # video banner (alarm)
        if s["alarm_active"]:
            self.video.set_banner(*self._banner(s))
        elif s["fall_state"] == "monitoring":
            rem = max(0, s["ground_alarm_seconds"] - s["seconds_on_ground"])
            self.video.set_banner(f"HEDEF YERDE — ALARM {rem:.0f} sn SONRA", WARN)
        elif s["target_state"] == "missing":
            rem = max(0.0, s["missing_alarm_seconds"] - (s["missing_elapsed"] or 0))
            self.video.set_banner(f"HEDEF KAYIP — ALARM {rem:.1f} sn SONRA", WARN)
        else:
            self.video.set_banner("", "")

        # tiles
        self.tile_people.set(s["person_count"], ACCENT if s["person_count"] else TEXT_FAINT)
        if s["target_id"] is None:
            self.tile_target.set("yok", TEXT_FAINT)
        elif s["target_state"] == "present":
            self.tile_target.set(f"#{s['target_id']}", OK)
        else:
            self.tile_target.set("kayıp", WARN)
        if s["alarm_active"]:
            self.tile_alarm.set({"fall": "DÜŞME", "fight": "KAVGA", "missing": "KAYIP"}.get(s["alarm_reason"], "AKTİF"),
                                DANGER if s["alarm_reason"] != "fight" else FIGHT)
        else:
            self.tile_alarm.set("—", TEXT_FAINT)

        # AI anahtarını gerçeğe eşitle (model yüklenemezse kendiliğinden kapanır)
        ai_on = bool(s.get("fight_ai_active"))
        if self.row_ai.toggle.isChecked() != ai_on:
            self.row_ai.toggle.blockSignals(True)
            self.row_ai.toggle.setChecked(ai_on)
            self.row_ai.toggle.blockSignals(False)

    @staticmethod
    def _headline(s):
        if s["alarm_active"]:
            r = s["alarm_reason"]
            return ({"fall": "DÜŞME ALARMI", "fight": "OLASI KAVGA", "missing": "KAYIP ALARMI"}
                    .get(r, "ALARM"), FIGHT if r == "fight" else DANGER)
        if s["fall_state"] == "monitoring" or s["target_state"] == "missing":
            return "İzleniyor — uyarı yakın", WARN
        if s["target_state"] == "present":
            return "Hedef takip ediliyor", OK
        return "Hazır — hedef seç", ACCENT

    @staticmethod
    def _banner(s):
        r = s["alarm_reason"]
        if r == "fall":
            return f"DÜŞME ALARMI  ·  {int(s['seconds_on_ground'])} sn yerde", DANGER
        if r == "fight":
            pairs = s.get("fighting_pairs", [])
            who = f"  ·  #{pairs[0][0]}–#{pairs[0][1]}" if pairs else ""
            tail = "AI onaylı" if s.get("fight_ai_active") else "düşük güven"
            return f"OLASI KAVGA{who}  ·  {tail}", FIGHT
        return f"KAYIP ALARMI  ·  {(s['missing_elapsed'] or 0):.0f} sn kadraj dışı", DANGER

    def _log_line(self, msg):
        ts = time.strftime("%H:%M:%S")
        item = QListWidgetItem(f"{ts}  {msg}")
        low = msg.lower()
        if "alarm" in low or "hata" in low or "düş" in low or "kavga" in low:
            item.setForeground(QColor(DANGER if "kavga" not in low else FIGHT))
        elif "hedef" in low or "hazır" in low or "tanındı" in low:
            item.setForeground(QColor(TEXT))
        else:
            item.setForeground(QColor(TEXT_DIM))
        self.log.addItem(item)
        self.log.scrollToBottom()
        if self.log.count() > 400:
            self.log.takeItem(0)

    def _on_error(self, msg):
        self._log_line(f"HATA: {msg}")
        self.hdr_state.setText("Hata")
        self.hdr_dot.set_color(DANGER)

    def closeEvent(self, ev):
        self._stop_worker()
        super().closeEvent(ev)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.resize(1420, 740)
    w.setMinimumSize(1200, 690)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
