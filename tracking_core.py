"""
tracking_core.py
==================
Çocuk Oyun Alanı - paylaşılan görüntü işleme / takip / güvenlik mantığı.

Bu modül arayüzden bağımsızdır: hem sade OpenCV penceresi kullanan
`playground_test.py` hem de modern masaüstü uygulaması `desktop_app.py`
bu mantığı kullanır.

Katmanlar:
  PersonDetector   - YOLOv8-pose ile kişi kutuları + iskelet noktaları
  SimpleTracker    - kare-kareye kimlik (konum + kıyafet rengi + beden oranı)
  TargetTracker    - seçilen hedefin kadraj dışına çıkma / yeniden tanınma durumu
  FallMonitor      - hedefin düşüp belli süre yerde kalması
  SafetyMonitor    - hepsini birleştirir, tek alarmı yönetir, tek bir durum döndürür
"""

import math
import time
import threading
import collections
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO


# ----------------------------- VARSAYILAN AYARLAR ----------------------------- #

DEFAULT_MODEL_NAME = "yolov8n-pose.pt"  # kişi kutusu + iskelet (COCO 17 nokta)
DEFAULT_CONF_THRESHOLD = 0.4            # kişi tespiti güven eşiği
DEFAULT_IOU_MATCH_THRESHOLD = 0.3       # kare-kareye takip eşleştirme eşiği
DEFAULT_MAX_LOST_FRAMES = 15            # bu kadar kare görünmezse track silinir
DEFAULT_REID_MATCH_THRESHOLD = 0.5      # kaybolan hedefi görünüm+beden oranıyla yeniden tanıma eşiği (0-1)
DEFAULT_MISSING_ALARM_SECONDS = 10      # hedef bu kadar saniye kadraj dışıysa alarm
DEFAULT_GROUND_ALARM_SECONDS = 60       # hedef bu kadar saniye (hareketsiz) yerde kalırsa alarm
DEFAULT_AGITATION_THRESHOLD = 1.9       # kavga: uzuv hızı eşiği (gövde-boyu/sn); düşük = daha hassas

BEEP_FREQ_HZ = 1000
BEEP_DURATION_MS = 150
BEEP_GAP_SECONDS = 0.35

# COCO iskelet nokta indeksleri (YOLOv8-pose)
KP_NOSE = 0
KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER = 5, 6
KP_LEFT_ELBOW, KP_RIGHT_ELBOW = 7, 8
KP_LEFT_WRIST, KP_RIGHT_WRIST = 9, 10
KP_LEFT_HIP, KP_RIGHT_HIP = 11, 12
KP_MIN_CONF = 0.3


# --------------------------- TEMEL VERİ YAPILARI --------------------------- #

@dataclass
class Detection:
    bbox: tuple                     # (x1, y1, x2, y2)
    keypoints: np.ndarray = None    # (17, 3) -> her satır (x, y, conf); yoksa None


@dataclass
class Track:
    track_id: int
    bbox: tuple
    lost_frames: int = 0
    histogram: np.ndarray = None    # kıyafet rengi imzası
    keypoints: np.ndarray = None    # son bilinen iskelet


# --------------------------- ALARM SESİ --------------------------- #

class AlarmPlayer:
    """Ayrı bir thread'de sürekli 'dıt dıt dıt' bipi çalar, durdurulana kadar."""

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _run(self):
        import winsound  # Windows'a özel, ekstra kurulum gerektirmez
        while not self._stop_event.is_set():
            winsound.Beep(BEEP_FREQ_HZ, BEEP_DURATION_MS)
            self._stop_event.wait(BEEP_GAP_SECONDS)


# --------------------------- GÖRÜNTÜ / GEOMETRİ YARDIMCILARI --------------------------- #

def iou(box_a, box_b):
    """İki kutu arasındaki kesişim/birleşim (IoU) oranı."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def compute_histogram(frame, bbox):
    """Bir kutunun içindeki bölgeden HSV renk histogramı çıkarır (görünüm imzası)."""
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
    h, w = frame.shape[:2]
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def histogram_similarity(hist_a, hist_b):
    """İki histogram arasındaki benzerlik (0-1, 1 = aynı)."""
    if hist_a is None or hist_b is None:
        return 0.0
    return max(0.0, cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def bbox_aspect_ratio(bbox):
    """
    Kutunun boy/en oranı (height / width). Kişinin bedensel oranının
    (uzun-ince mi, kısa-geniş mi) kabaca kameradan uzaklıktan bağımsız
    bir özeti — iki kişi aynı mesafede olmasa da oran genelde korunur.
    """
    x1, y1, x2, y2 = bbox
    width = max(1e-6, x2 - x1)
    height = max(1e-6, y2 - y1)
    return height / width


def shape_similarity(ratio_a, ratio_b):
    """İki boy/en oranı arasındaki benzerlik (0-1, 1 = aynı oran)."""
    if ratio_a is None or ratio_b is None:
        return 0.0
    largest = max(ratio_a, ratio_b, 1e-6)
    diff = abs(ratio_a - ratio_b)
    return max(0.0, 1.0 - (diff / largest))


def classify_posture(track):
    """
    Kişinin duruşunu kabaca sınıflar: "fallen" | "upright" | "unknown".

    Önce iskeleti kullanır: omuz merkezinden kalça merkezine giden gövde
    ekseni yataya yakınsa (açı küçük) -> yatık = "fallen"; dikeyse -> "upright".
    İskelet güvenilir değilse kutunun boy/en oranına düşer.
    """
    kp = track.keypoints
    if kp is not None and len(kp) > KP_RIGHT_HIP:
        def pt(i):
            x, y, c = kp[i]
            return (float(x), float(y)) if c >= KP_MIN_CONF else None

        shoulders = [p for p in (pt(KP_LEFT_SHOULDER), pt(KP_RIGHT_SHOULDER)) if p is not None]
        hips = [p for p in (pt(KP_LEFT_HIP), pt(KP_RIGHT_HIP)) if p is not None]
        if shoulders and hips:
            sx = sum(p[0] for p in shoulders) / len(shoulders)
            sy = sum(p[1] for p in shoulders) / len(shoulders)
            hx = sum(p[0] for p in hips) / len(hips)
            hy = sum(p[1] for p in hips) / len(hips)
            dx = abs(sx - hx)
            dy = abs(sy - hy)
            angle = math.degrees(math.atan2(dy, dx + 1e-6))  # 90 = dik gövde, 0 = yatık
            if angle <= 35.0:
                return "fallen"
            if angle >= 55.0:
                return "upright"

    ratio = bbox_aspect_ratio(track.bbox)
    if ratio <= 0.85:
        return "fallen"
    if ratio >= 1.5:
        return "upright"
    return "unknown"


# --------------------------- KİŞİ TESPİTİ (YOLOv8-POSE) --------------------------- #

class PersonDetector:
    """YOLOv8-pose ile kişi kutuları + iskelet noktaları döndürür."""

    def __init__(self, model_name=DEFAULT_MODEL_NAME, conf_threshold=DEFAULT_CONF_THRESHOLD):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints

        kp_xy = None
        kp_conf = None
        if keypoints is not None and keypoints.xy is not None:
            kp_xy = keypoints.xy.cpu().numpy()          # (N, 17, 2)
            if keypoints.conf is not None:
                kp_conf = keypoints.conf.cpu().numpy()  # (N, 17)

        detections = []
        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            kp = None
            if kp_xy is not None and i < len(kp_xy):
                xy = kp_xy[i]                                   # (17, 2)
                c = kp_conf[i] if kp_conf is not None else np.ones(len(xy))
                kp = np.concatenate([xy, c[:, None]], axis=1)   # (17, 3)

            detections.append(Detection(bbox=(x1, y1, x2, y2), keypoints=kp))
        return detections


# --------------------------- BASİT ÇOK-KİŞİLİ TAKİPÇİ --------------------------- #

class SimpleTracker:
    """
    Konum (IoU) + görünüm (renk histogramı) + beden oranı (boy/en) birlikte
    kullanan, hafif bir çok-nesne takipçisi. Amaç: kare kareye aynı kişiye
    aynı ID'yi vermek (ağır bir derin ReID modeli olmadan, test için yeterli).

    Üç sinyal birlikte kullanılır çünkü sadece konuma (IoU) bakan eşleştirme,
    birkaç kişi yan yana durup kutuları çakıştığında yanlış kişiye "aynı hedef"
    diyebiliyordu. TÜM aday (track, tespit) çiftleri skorlanıp en iyiden
    başlanarak açgözlü atanır.
    """

    IOU_WEIGHT = 0.5
    APPEARANCE_WEIGHT = 0.3
    SHAPE_WEIGHT = 0.2

    def __init__(self, iou_match_threshold=DEFAULT_IOU_MATCH_THRESHOLD,
                 max_lost_frames=DEFAULT_MAX_LOST_FRAMES):
        self.iou_match_threshold = iou_match_threshold
        self.max_lost_frames = max_lost_frames
        self._next_id = 1
        self.tracks = {}  # {track_id: Track}

    def update(self, frame, detections):
        """
        detections: [Detection, ...] bu karede bulunan kişiler.
        Döndürür: {det_idx: yeni_track_id} - önceki hiçbir track ile
        eşleşmeyip yeni oluşturulan tespitler (re-id için kullanılır).
        """
        detection_histograms = [compute_histogram(frame, d.bbox) for d in detections]

        candidates = []
        for track_id, track in self.tracks.items():
            for det_idx, det in enumerate(detections):
                iou_score = iou(track.bbox, det.bbox)
                if iou_score < self.iou_match_threshold:
                    continue
                appearance_score = histogram_similarity(track.histogram, detection_histograms[det_idx])
                shape_score = shape_similarity(bbox_aspect_ratio(track.bbox), bbox_aspect_ratio(det.bbox))
                combined = (
                    iou_score * self.IOU_WEIGHT
                    + appearance_score * self.APPEARANCE_WEIGHT
                    + shape_score * self.SHAPE_WEIGHT
                )
                candidates.append((combined, track_id, det_idx))

        candidates.sort(key=lambda c: c[0], reverse=True)
        matched_track_ids = set()
        matched_det_indices = set()
        for combined, track_id, det_idx in candidates:
            if track_id in matched_track_ids or det_idx in matched_det_indices:
                continue
            matched_track_ids.add(track_id)
            matched_det_indices.add(det_idx)
            track = self.tracks[track_id]
            track.bbox = detections[det_idx].bbox
            track.keypoints = detections[det_idx].keypoints
            track.lost_frames = 0
            track.histogram = detection_histograms[det_idx]

        for track_id, track in self.tracks.items():
            if track_id not in matched_track_ids:
                track.lost_frames += 1
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id].lost_frames > self.max_lost_frames:
                del self.tracks[track_id]

        new_track_ids_for_detections = {}
        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_indices:
                continue
            track = Track(
                track_id=self._next_id,
                bbox=det.bbox,
                histogram=detection_histograms[det_idx],
                keypoints=det.keypoints,
            )
            self.tracks[self._next_id] = track
            new_track_ids_for_detections[det_idx] = self._next_id
            self._next_id += 1

        return new_track_ids_for_detections


# --------------------------- HEDEF (KADRAJ DIŞI / YENİDEN TANIMA) --------------------------- #

class TargetTracker:
    """
    Seçilen hedefin durumu: "no_target" | "present" | "missing" | "alarm".
    ("alarm" = kadraj dışı süresi eşiği aştı; sesi SafetyMonitor yönetir.)

    Yeniden tanıma skoru kıyafet rengi + boy/en oranı birlikte hesaplanır.
    """

    REID_APPEARANCE_WEIGHT = 0.65
    REID_SHAPE_WEIGHT = 0.35

    def __init__(self, missing_alarm_seconds=DEFAULT_MISSING_ALARM_SECONDS,
                 reid_match_threshold=DEFAULT_REID_MATCH_THRESHOLD):
        self.missing_alarm_seconds = missing_alarm_seconds
        self.reid_match_threshold = reid_match_threshold

        self.target_id = None
        self.target_histogram = None
        self.target_shape_ratio = None
        self.missing_since = None

        self.on_event = None

    def _log(self, message):
        if self.on_event:
            self.on_event(message)

    def select(self, track):
        self.target_id = track.track_id
        self.target_histogram = track.histogram
        self.target_shape_ratio = bbox_aspect_ratio(track.bbox)
        self.missing_since = None
        self._log(f"Hedef seçildi (ID {track.track_id})")

    def reset(self):
        if self.target_id is not None:
            self._log("Hedef seçimi sıfırlandı")
        self.target_id = None
        self.target_histogram = None
        self.target_shape_ratio = None
        self.missing_since = None

    def update(self, tracker, new_track_ids_for_detections):
        if self.target_id is None:
            return "no_target"

        if self.target_id in tracker.tracks:
            current = tracker.tracks[self.target_id]
            self.target_histogram = current.histogram
            self.target_shape_ratio = bbox_aspect_ratio(current.bbox)
            if self.missing_since is not None:
                self._log("Hedef yeniden görüldü (aynı ID)")
            self.missing_since = None
            return "present"

        if self.missing_since is None:
            self.missing_since = time.time()
            self._log("Hedef kadrajdan kayboldu, sayaç başladı")

        for det_idx, new_id in new_track_ids_for_detections.items():
            candidate = tracker.tracks[new_id]
            appearance_score = histogram_similarity(self.target_histogram, candidate.histogram)
            shape_score = shape_similarity(self.target_shape_ratio, bbox_aspect_ratio(candidate.bbox))
            combined = (
                appearance_score * self.REID_APPEARANCE_WEIGHT
                + shape_score * self.REID_SHAPE_WEIGHT
            )
            if combined >= self.reid_match_threshold:
                self._log(
                    f"Hedef yeniden tanındı (ID {self.target_id} -> {new_id}, "
                    f"benzerlik {combined:.2f})"
                )
                self.target_id = new_id
                self.target_histogram = candidate.histogram
                self.target_shape_ratio = bbox_aspect_ratio(candidate.bbox)
                self.missing_since = None
                return "present"

        elapsed = time.time() - self.missing_since
        if elapsed >= self.missing_alarm_seconds:
            return "alarm"
        return "missing"


# --------------------------- DÜŞME TESPİTİ --------------------------- #

class FallMonitor:
    """
    Hedefin düşüp belli süre (varsayılan 60 sn) YERDE KALMASINI izler.

    Durumlar: "upright" | "monitoring" | "alarm"
      upright     - hedef ayakta / oturuyor / normal
      monitoring  - hedef yatık duruşta, yerde kalma süresi ölçülüyor
      alarm       - hedef eşik süre kadar (hareketsiz) yerde kaldı

    Mantık:
      - Kısa bir doğrulama süresi (FALL_DEBOUNCE) boyunca sürekli "yatık"
        görülürse "düştü" sayılır ve sayaç başlar.
      - Hedef belirgin şekilde hareket ediyorsa (yerde yuvarlanıyor, oynuyor)
        sayaç ilerlemez — "kalkamıyor" değil "oynuyor" kabul edilir.
      - Hedef ayağa kalkarsa (STAND_DEBOUNCE boyunca "ayakta") sayaç sıfırlanır.
      - Birikmiş hareketsiz-yerde süresi eşiği aşınca "alarm".
    """

    FALL_DEBOUNCE_SECONDS = 1.2
    STAND_DEBOUNCE_SECONDS = 1.5
    MOTION_PAUSE_THRESHOLD = 0.18  # gövde-boyu / saniye; üstü = "hareketli"

    def __init__(self, ground_seconds=DEFAULT_GROUND_ALARM_SECONDS):
        self.ground_seconds = ground_seconds
        self.on_event = None
        self.reset()

    def reset(self):
        self.state = "upright"
        self.ground_accum = 0.0
        self._fallen_streak_start = None
        self._upright_streak_start = None
        self._prev_centroid = None
        self._last_time = None

    def _log(self, message):
        if self.on_event:
            self.on_event(message)

    def _estimate_motion(self, track, dt):
        x1, y1, x2, y2 = track.bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        height = max(1.0, y2 - y1)
        if self._prev_centroid is None or dt <= 0:
            self._prev_centroid = (cx, cy)
            return 0.0
        dist = math.hypot(cx - self._prev_centroid[0], cy - self._prev_centroid[1])
        self._prev_centroid = (cx, cy)
        return dist / height / dt

    def update(self, target_track):
        """target_track: hedefin bu karedeki Track'i, görünmüyorsa None."""
        now = time.time()

        if target_track is None:
            # Hedef görünmüyor: sayaç dursun, mevcut durum korunsun.
            self._prev_centroid = None
            self._last_time = None
            self._fallen_streak_start = None
            self._upright_streak_start = None
            return self.state

        dt = 0.0 if self._last_time is None else max(0.0, now - self._last_time)
        self._last_time = now

        posture = classify_posture(target_track)
        motion = self._estimate_motion(target_track, dt)

        if posture == "fallen":
            self._upright_streak_start = None
            if self._fallen_streak_start is None:
                self._fallen_streak_start = now
            if self.state == "upright" and (now - self._fallen_streak_start) >= self.FALL_DEBOUNCE_SECONDS:
                self.state = "monitoring"
                self.ground_accum = 0.0
                self._log("Hedef yere düştü — yerde kalma süresi ölçülüyor")
        elif posture == "upright":
            self._fallen_streak_start = None
            if self._upright_streak_start is None:
                self._upright_streak_start = now
            if self.state in ("monitoring", "alarm") and \
                    (now - self._upright_streak_start) >= self.STAND_DEBOUNCE_SECONDS:
                self._log("Hedef ayağa kalktı — düşme takibi kapandı")
                self.state = "upright"
                self.ground_accum = 0.0
        # posture == "unknown": streak'lere dokunma, mevcut durumu koru

        if self.state == "monitoring":
            if motion <= self.MOTION_PAUSE_THRESHOLD:
                self.ground_accum += dt
            if self.ground_accum >= self.ground_seconds:
                self.state = "alarm"
                self._log(f"DÜŞME ALARMI — hedef {int(self.ground_seconds)} sn yerden kalkamadı")

        return self.state


# --------------------------- KAVGA / İTİŞ-KAKIŞ TESPİTİ (DENEYSEL) --------------------------- #

class FightMonitor:
    """
    Kadrajdaki HERHANGİ İKİ kişi arasında olası kavga/itiş-kakış izler.
    DENEYSEL ve DÜŞÜK GÜVEN: güreşme / şakalaşarak itişme gerçek kavgaya
    çok benzer, yanlış alarm olabilir. "Bir bak" seviyesinde bir uyarıdır.

    Sinyaller (her kişi çifti için):
      1. Temas       - kutular çakışıyor (IoU) ya da merkez mesafesi gövde
                       boyuna göre çok yakın.
      2. Ajitasyon   - iki kişinin de bilek/dirsek noktaları hızlı hareket
                       ediyor (gövde-boyu/sn). İskelet yoksa kutu merkezinin
                       hızına düşer.
      3. Süreklilik  - temas + yüksek ajitasyon FIGHT_DEBOUNCE boyunca
                       kesintisiz sürerse o çift "alarm"a geçer.
      Sakinleşince (CALM_DEBOUNCE) ya da temas kopunca çift temizlenir.

    Durumlar: "calm" | "watching" | "alarm"
    """

    CONTACT_IOU_THRESHOLD = 0.10
    CONTACT_DIST_FACTOR = 0.85        # merkez mesafesi < faktör * ort. gövde yüksekliği
    MIN_BOTH_ACTIVITY = 0.6           # iki kişi de en az bu kadar hareketli olmalı
    FIGHT_DEBOUNCE_SECONDS = 2.0
    CALM_DEBOUNCE_SECONDS = 3.0
    VELOCITY_WINDOW_SECONDS = 0.5
    HISTORY_MAXLEN = 40
    SCENE_SAMPLE_INTERVAL = 0.12      # AI klibi için sahne karesi örnekleme (~8 fps)
    SCENE_REQUEST_INTERVAL = 0.5      # AI'a en sık bu aralıkla klip gönder
    SCENE_FRAME_WIDTH = 320           # AI klibi için küçültülmüş kare genişliği

    def __init__(self, agitation_threshold=DEFAULT_AGITATION_THRESHOLD, enabled=True,
                 scorer=None, ai_threshold=0.55):
        self.agitation_threshold = agitation_threshold
        self.enabled = enabled
        self.scorer = scorer          # violence_model.AsyncViolenceScorer veya None
        self.use_ai = scorer is not None
        self.ai_threshold = ai_threshold
        self.on_event = None
        self._history = {}   # track_id -> deque[(t, bbox, keypoints)]
        self._pairs = {}     # frozenset({id_a, id_b}) -> pair-state dict
        self.fighting_pairs = []  # [(id_a, id_b), ...] şu an "alarm" olan çiftler
        self.last_ai_score = None
        self._scene_buffer = collections.deque(
            maxlen=(scorer.clip_len if scorer is not None else 8))
        self._last_scene_sample = 0.0
        self._last_scene_request = 0.0

    def _log(self, message):
        if self.on_event:
            self.on_event(message)

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _in_contact(self, ta, tb):
        if iou(ta.bbox, tb.bbox) >= self.CONTACT_IOU_THRESHOLD:
            return True
        ca, cb = self._center(ta.bbox), self._center(tb.bbox)
        ha = ta.bbox[3] - ta.bbox[1]
        hb = tb.bbox[3] - tb.bbox[1]
        mean_h = max(1.0, (ha + hb) / 2.0)
        return math.hypot(ca[0] - cb[0], ca[1] - cb[1]) <= self.CONTACT_DIST_FACTOR * mean_h

    def _activity(self, hist):
        """
        Son ~0.5 sn'de en hareketli uzvun (bilek/dirsek) YOL UZUNLUĞU / süre
        (gövde-boyu/sn). Yol uzunluğu kullanılır çünkü kavga hareketi salınımlı;
        sadece baş-son farkına bakmak sallanmayı sıfır sanabilir. İskelet yoksa
        kutu merkezinin yol uzunluğuna düşer.
        """
        if len(hist) < 2:
            return 0.0
        t_new = hist[-1][0]
        window = [s for s in hist if t_new - s[0] <= self.VELOCITY_WINDOW_SECONDS]
        if len(window) < 2:
            return 0.0
        total_dt = window[-1][0] - window[0][0]
        if total_dt <= 1e-3:
            return 0.0
        height = max(1.0, window[-1][1][3] - window[-1][1][1])

        best = 0.0
        for idx in (KP_LEFT_ELBOW, KP_RIGHT_ELBOW, KP_LEFT_WRIST, KP_RIGHT_WRIST):
            path = 0.0
            counted = False
            for (_, _, kp_p), (_, _, kp_q) in zip(window, window[1:]):
                if kp_p is None or kp_q is None or idx >= len(kp_p) or idx >= len(kp_q):
                    continue
                xp, yp, cp = kp_p[idx]
                xq, yq, cq = kp_q[idx]
                if cp >= KP_MIN_CONF and cq >= KP_MIN_CONF:
                    path += math.hypot(xq - xp, yq - yp)
                    counted = True
            if counted:
                best = max(best, path / height / total_dt)
        if best > 0.0:
            return best

        path = 0.0
        for (_, bbox_p, _), (_, bbox_q, _) in zip(window, window[1:]):
            cp = self._center(bbox_p)
            cq = self._center(bbox_q)
            path += math.hypot(cq[0] - cp[0], cq[1] - cp[1])
        return path / height / total_dt

    def update(self, tracks, frame=None):
        """
        tracks: {track_id: Track} — bu karedeki tüm kişiler.
        frame:  BGR kare (AI doğrulaması için; None ise sadece sezgisel).
        Genel durumu döndürür.
        """
        now = time.time()

        for tid, track in tracks.items():
            hist = self._history.setdefault(
                tid, collections.deque(maxlen=self.HISTORY_MAXLEN))
            hist.append((now, track.bbox, track.keypoints))
        for tid in list(self._history.keys()):
            if tid not in tracks:
                del self._history[tid]

        if not self.enabled:
            self._pairs.clear()
            self._scene_buffer.clear()
            self.fighting_pairs = []
            self.last_ai_score = None
            return "calm"

        ai_on = self.scorer is not None and self.use_ai and frame is not None

        # AI klibi için sahne karelerini sürekli topla (fırtına başlamadan hazır olsun)
        if ai_on and now - self._last_scene_sample >= self.SCENE_SAMPLE_INTERVAL:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (self.SCENE_FRAME_WIDTH,
                                       max(1, int(self.SCENE_FRAME_WIDTH * h / w))))
            self._scene_buffer.append(small)
            self._last_scene_sample = now

        scene_score = self.scorer.get_fresh("scene") if ai_on else None
        self.last_ai_score = scene_score

        ids = list(tracks.keys())
        seen_keys = set()
        any_alarm = False
        any_watching = False
        fighting_pairs = []

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                ta, tb = tracks[a], tracks[b]
                if not self._in_contact(ta, tb):
                    continue

                key = frozenset((a, b))
                seen_keys.add(key)
                pair = self._pairs.get(key)
                if pair is None:
                    pair = {"fight_like_since": None, "last_fight_like": 0.0, "state": "watching"}
                    self._pairs[key] = pair

                act_a = self._activity(self._history.get(a, ()))
                act_b = self._activity(self._history.get(b, ()))
                both_active = min(act_a, act_b) >= self.MIN_BOTH_ACTIVITY
                agitation = (act_a + act_b) / 2.0
                heuristic_fight_like = both_active and agitation >= self.agitation_threshold

                # AI doğrulaması açıksa: sahne modeli "şiddet" demedikçe alarm yok.
                if ai_on:
                    if heuristic_fight_like and len(self._scene_buffer) >= self._scene_buffer.maxlen \
                            and now - self._last_scene_request >= self.SCENE_REQUEST_INTERVAL:
                        self.scorer.request("scene", list(self._scene_buffer))
                        self._last_scene_request = now
                    fight_like = (
                        heuristic_fight_like
                        and scene_score is not None
                        and scene_score >= self.ai_threshold
                    )
                else:
                    fight_like = heuristic_fight_like

                if fight_like:
                    pair["last_fight_like"] = now
                    if pair["fight_like_since"] is None:
                        pair["fight_like_since"] = now
                    if pair["state"] != "alarm" and \
                            (now - pair["fight_like_since"]) >= self.FIGHT_DEBOUNCE_SECONDS:
                        pair["state"] = "alarm"
                        self._log(f"OLASI KAVGA — #{min(a, b)} ile #{max(a, b)} arasında yoğun hareket")
                else:
                    pair["fight_like_since"] = None
                    if pair["state"] == "alarm" and \
                            (now - pair["last_fight_like"]) >= self.CALM_DEBOUNCE_SECONDS:
                        pair["state"] = "watching"
                        self._log(f"#{min(a, b)} ile #{max(a, b)} sakinleşti")

                if pair["state"] == "alarm":
                    any_alarm = True
                    fighting_pairs.append((min(a, b), max(a, b)))
                else:
                    any_watching = True

        for key in list(self._pairs.keys()):
            if key not in seen_keys:
                del self._pairs[key]

        self.fighting_pairs = fighting_pairs
        if any_alarm:
            return "alarm"
        return "watching" if any_watching else "calm"


# --------------------------- HEPSİNİ BİRLEŞTİREN KATMAN --------------------------- #

class SafetyMonitor:
    """
    Tek giriş noktası: her kare için `process(frame)` çağır, durum sözlüğü al.
    Tespit + takip + kadraj-dışı + düşme + kavga mantığını birleştirir ve TEK
    alarmı yönetir (kadraj dışı VEYA düşme VEYA kavga -> alarm çalar).
    """

    def __init__(self, missing_alarm_seconds=DEFAULT_MISSING_ALARM_SECONDS,
                 ground_alarm_seconds=DEFAULT_GROUND_ALARM_SECONDS,
                 fight_enabled=True, use_violence_ai=False, violence_kind="movinet",
                 model_name=DEFAULT_MODEL_NAME):
        self._model_name = model_name
        self._use_violence_ai = use_violence_ai
        self._violence_kind = violence_kind
        self.detector = None  # ilk kullanımda (tercihen worker thread'inde) yüklenir
        self.violence_classifier = None
        self.tracker = SimpleTracker()
        self.target = TargetTracker(missing_alarm_seconds=missing_alarm_seconds)
        self.fall = FallMonitor(ground_seconds=ground_alarm_seconds)
        self.fight = FightMonitor(enabled=fight_enabled)

        self._alarm = AlarmPlayer()
        self.alarm_active = False
        self.alarm_reason = None  # "missing" | "fall" | "fight" | None
        self._on_event = None

    @property
    def on_event(self):
        return self._on_event

    @on_event.setter
    def on_event(self, callback):
        self._on_event = callback
        self.target.on_event = callback
        self.fall.on_event = callback
        self.fight.on_event = callback

    def _emit(self, message):
        if self._on_event:
            self._on_event(message)

    def load_model(self):
        if self.detector is None:
            self.detector = PersonDetector(model_name=self._model_name)
        if self._use_violence_ai and self.fight.scorer is None:
            self._load_violence_ai()

    def _load_violence_ai(self):
        try:
            from violence_model import create_violence_classifier, AsyncViolenceScorer
            self.violence_classifier = create_violence_classifier(self._violence_kind)
            self.violence_classifier.on_event = self._emit
            self.violence_classifier.load()
            self.fight.scorer = AsyncViolenceScorer(self.violence_classifier)
            self.fight._scene_buffer = collections.deque(maxlen=self.fight.scorer.clip_len)
            self.fight.use_ai = True
        except Exception as exc:
            self._emit(f"Kavga modeli yüklenemedi, sadece sezgisel kavga tespiti: {exc}")
            self.fight.scorer = None
            self.fight.use_ai = False

    def set_violence_ai_enabled(self, enabled):
        """Çalışırken AI doğrulamasını aç/kapat (model yüklüyse)."""
        if enabled and self.fight.scorer is None:
            self._use_violence_ai = True
            self._load_violence_ai()
        self.fight.use_ai = bool(enabled) and self.fight.scorer is not None

    def select_target_at(self, x, y):
        """Verilen ekran noktasında bir kişi varsa hedef seçer. Başarılıysa True."""
        for track in self.tracker.tracks.values():
            x1, y1, x2, y2 = track.bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.target.select(track)
                self.fall.reset()
                return True
        return False

    def reset_target(self):
        self.target.reset()
        self.fall.reset()

    def set_fight_enabled(self, enabled):
        self.fight.enabled = bool(enabled)

    def set_fight_sensitivity(self, threshold):
        """Düşük eşik = daha hassas (daha kolay kavga der)."""
        self.fight.agitation_threshold = float(threshold)

    def process(self, frame):
        if self.detector is None:
            self.load_model()

        detections = self.detector.detect(frame)
        new_ids = self.tracker.update(frame, detections)
        target_state = self.target.update(self.tracker, new_ids)

        target_track = None
        if target_state == "present" and self.target.target_id in self.tracker.tracks:
            target_track = self.tracker.tracks[self.target.target_id]
        fall_state = self.fall.update(target_track)
        fight_state = self.fight.update(self.tracker.tracks, frame)

        fall_alarm = (fall_state == "alarm")
        missing_alarm = (target_state == "alarm")
        fight_alarm = (fight_state == "alarm")
        should_alarm = fall_alarm or missing_alarm or fight_alarm
        if fall_alarm:
            reason = "fall"
        elif fight_alarm:
            reason = "fight"
        elif missing_alarm:
            reason = "missing"
        else:
            reason = None

        if should_alarm and not self.alarm_active:
            self._alarm.start()
            self.alarm_active = True
            self._emit("ALARM başladı")
        elif not should_alarm and self.alarm_active:
            self._alarm.stop()
            self.alarm_active = False
            self._emit("Alarm durduruldu")
        self.alarm_reason = reason if self.alarm_active else None

        missing_elapsed = None
        if self.target.missing_since is not None:
            missing_elapsed = time.time() - self.target.missing_since

        return {
            "target_state": target_state,
            "target_id": self.target.target_id,
            "missing_elapsed": missing_elapsed,
            "missing_alarm_seconds": self.target.missing_alarm_seconds,
            "fall_state": fall_state,
            "seconds_on_ground": self.fall.ground_accum,
            "ground_alarm_seconds": self.fall.ground_seconds,
            "fight_state": fight_state,
            "fight_enabled": self.fight.enabled,
            "fight_ai_active": self.fight.scorer is not None and self.fight.use_ai,
            "fight_last_ai_score": self.fight.last_ai_score,
            "fighting_pairs": list(self.fight.fighting_pairs),
            "person_count": len(self.tracker.tracks),
            "alarm_active": self.alarm_active,
            "alarm_reason": self.alarm_reason,
        }

    def shutdown(self):
        self._alarm.stop()
        self.alarm_active = False
        if self.fight.scorer is not None:
            self.fight.scorer.stop()
