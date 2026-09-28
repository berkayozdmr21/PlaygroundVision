"""
violence_model.py
==================
Hazır (önceden eğitilmiş) şiddet/kavga sınıflandırıcısı sarmalayıcıları.

MİMARİ ROLÜ
  `FightMonitor` sezgisel olarak (temas + hızlı hareket) bir kişi çifti
  yakaladığında, sahnenin son ~1 saniyesi buradaki modele "gerçekten
  şiddet mi?" diye sorulur. Sezgisel + model ikisi birden "evet" derse
  alarm — yanlış alarm azalır. Model AYRI thread'de çalışır
  (`AsyncViolenceScorer`), video akışı donmaz.

İKİ MODEL SEÇENEĞİ
  1) "movinet" (ÖNERİLEN, varsayılan) — VD-MIL / MoViNet-A0, RWF-2000 +
     SCVD ile eğitilmiş, kavga + silah odaklı, CPU'da çalışır, 8 karelik
     klip alır. Ağırlıklar `models/vdmil/` içinde.
     LİSANS: CC BY-NC 4.0 — SADECE TİCARİ OLMAYAN kullanım (test/prototip).
     Kurulum: pip install git+https://github.com/roggerfq/MoViNet-pytorch.git
  2) "vit" — tek kare ViT (RLVS). Kurulumu kolay ama kare-bazlı ve
     yetişkin sokak-kavgası verisiyle eğitilmiş; kalitesi düşük.
     Kurulum: pip install -U "transformers>=4.45" pillow

Sınıflandırıcı arayüzü (ikisi de sağlar):
  .clip_len            -> kaç kare beklediği (movinet: 8, vit: 1)
  .load()             -> modeli belleğe alır
  .ready              -> yüklendi mi
  .score(frames_bgr)  -> 0..1  (P(şiddet)); frames_bgr = BGR kare listesi
"""

import os
import time
import threading

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
VDMIL_DIR = os.path.join(_HERE, "models", "vdmil")

DEFAULT_VIT_MODEL = "jaranohaal/vit-base-violence-detection"
_VIOLENT_HINTS = ("viol", "fight", "kavga", "assault", "abuse")
_NONVIOLENT_HINTS = ("non", "no_", "no-", "normal", "safe", "peace", "calm")


# --------------------------- MoViNet (VD-MIL) --------------------------- #

class MoViNetViolenceClassifier:
    """
    VD-MIL (roggerfq) — MoViNet-A0 omurga + MIL ile eğitilmiş ikili baş.
    8 karelik BGR klip alır, 0..1 şiddet olasılığı döndürür.
    CC BY-NC 4.0 — ticari olmayan kullanım.
    """

    clip_len = 8

    def __init__(self, weights_dir=VDMIL_DIR, device="cpu"):
        self.weights_dir = weights_dir
        self.device = device
        self._clf = None
        self.on_event = None

    def _log(self, message):
        if self.on_event:
            self.on_event(message)

    @property
    def ready(self):
        return self._clf is not None

    def load(self):
        if self._clf is not None:
            return
        import sys
        if self.weights_dir not in sys.path:
            sys.path.insert(0, self.weights_dir)
        self._log("Kavga modeli (MoViNet/VD-MIL) yükleniyor...")
        from vdmil_movinet_classifier import MovinetClassifier  # models/vdmil/ içinde
        movinet_dir = os.path.join(self.weights_dir, "movinet_weights")
        head = os.path.join(self.weights_dir, "model_20.pt")
        if not os.path.exists(head):
            raise FileNotFoundError(f"Eğitilmiş baş bulunamadı: {head}")
        self._clf = MovinetClassifier(movinet_dir, [head], device=self.device)
        self._log("Kavga modeli hazır (MoViNet/VD-MIL).")

    def score(self, frames_bgr):
        if self._clf is None or not frames_bgr:
            return 0.0
        frames = list(frames_bgr)
        while len(frames) < self.clip_len:
            frames.insert(0, frames[0])
        frames = frames[-self.clip_len:]
        try:
            out = self._clf([frames])          # numpy (1, 1)
            return float(np.asarray(out).reshape(-1)[0])
        except Exception:
            return 0.0


# --------------------------- Tek kare ViT --------------------------- #

class ViolenceClassifier:
    """Tek kare image-classification modeli (transformers). Yedek seçenek."""

    clip_len = 1

    def __init__(self, model_name=DEFAULT_VIT_MODEL):
        self.model_name = model_name
        self._pipe = None
        self.labels = []
        self.on_event = None

    def _log(self, message):
        if self.on_event:
            self.on_event(message)

    @property
    def ready(self):
        return self._pipe is not None

    def load(self):
        if self._pipe is not None:
            return
        from transformers import pipeline
        self._log("Kavga modeli (ViT) yükleniyor...")
        self._pipe = pipeline("image-classification", model=self.model_name)
        try:
            self.labels = list(self._pipe.model.config.id2label.values())
        except Exception:
            self.labels = []
        self._log(f"Kavga modeli hazır (ViT). Etiketler: {self.labels}")

    def score(self, frames_bgr):
        if self._pipe is None or not frames_bgr:
            return 0.0
        from PIL import Image
        rgb = cv2.cvtColor(frames_bgr[-1], cv2.COLOR_BGR2RGB)
        results = self._pipe(Image.fromarray(rgb))
        scored = {r["label"].lower(): float(r["score"]) for r in results}
        for label, s in scored.items():
            if any(h in label for h in _VIOLENT_HINTS):
                return s
        for label, s in scored.items():
            if any(h in label for h in _NONVIOLENT_HINTS):
                return 1.0 - s
        for label, s in scored.items():
            if label in ("1", "label_1", "class_1"):
                return s
        return 0.0


def create_violence_classifier(kind="movinet"):
    kind = (kind or "movinet").lower()
    if kind == "vit":
        return ViolenceClassifier()
    return MoViNetViolenceClassifier()


# --------------------------- Async çalıştırıcı --------------------------- #

class AsyncViolenceScorer:
    """
    Sınıflandırıcıyı AYRI thread'de çalıştırır (CPU'da çıkarım yavaş olabilir).
    Ana döngü `request(key, frames)` ile en son klibi bırakır, sonucu hazır
    olunca `get_fresh(key)` ile okur.
    """

    def __init__(self, classifier, stale_after=2.0):
        self.classifier = classifier
        self.clip_len = getattr(classifier, "clip_len", 1)
        self.stale_after = stale_after
        self._lock = threading.Lock()
        self._pending = None
        self._scores = {}
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def request(self, key, frames):
        if not frames:
            return
        with self._lock:
            self._pending = (key, [f.copy() for f in frames])

    def get_fresh(self, key):
        entry = self._scores.get(key)
        if entry is None:
            return None
        ts, score = entry
        if time.time() - ts > self.stale_after:
            return None
        return score

    def forget(self, key):
        self._scores.pop(key, None)

    def stop(self):
        self._stop = True

    def _run(self):
        while not self._stop:
            with self._lock:
                job = self._pending
                self._pending = None
            if job is None:
                time.sleep(0.02)
                continue
            key, frames = job
            try:
                score = self.classifier.score(frames)
            except Exception:
                score = 0.0
            self._scores[key] = (time.time(), score)
