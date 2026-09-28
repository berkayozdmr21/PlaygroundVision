"""
clip_recorder.py
================
Test sırasında kameradan kısa video klipleri kaydeder — sonradan kendi
kavga tespiti modelini eğitmek için veri toplamak amacıyla.

Kullanım:
  rec = ClipRecorder(fps=15)
  ...her karede:
  rec.add(temiz_kare)            # kutu/iskelet çizilmemiş HAM kare
  ...
  rec.save("kavga")             # son ~8 sn + sonraki ~3 sn -> dataset/kavga/<zaman>.mp4
  rec.save("normal")            # "kavga değil ama benziyor" örnekleri

Klasör yapısı:
  dataset/
    kavga/     20260907_141230.mp4
    normal/    20260907_141412.mp4
    <etiket>/  ...

Her klibin yanına bir .txt (kayıt anındaki durum/olay) da yazılabilir.
"""

import os
import time
import collections
from datetime import datetime

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_DIR = os.path.join(_HERE, "dataset")


class ClipRecorder:
    def __init__(self, fps=15.0, pre_seconds=8.0, post_seconds=3.0,
                 max_width=720, out_root=DEFAULT_DATASET_DIR):
        self.fps = max(1.0, float(fps))
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.max_width = max_width          # tamponda RAM için kareleri küçült
        self.out_root = out_root
        self._buffer = collections.deque(maxlen=int(self.fps * (pre_seconds + 1)))
        self._writer = None
        self._writer_deadline = 0.0
        self._writer_path = None
        self.on_event = None

    def _fit(self, frame):
        h, w = frame.shape[:2]
        if w <= self.max_width:
            return frame
        return cv2.resize(frame, (self.max_width, max(1, int(h * self.max_width / w))))

    def _log(self, message):
        if self.on_event:
            self.on_event(message)

    @property
    def recording(self):
        return self._writer is not None

    def add(self, frame):
        """Her karede çağır (HAM kare — üzerine çizim yapılmamış olması iyi)."""
        fitted = self._fit(frame)
        self._buffer.append(fitted.copy() if fitted is frame else fitted)
        if self._writer is not None:
            self._writer.write(fitted)
            if time.time() >= self._writer_deadline:
                self._finish()

    def save(self, label="unlabeled", note=""):
        """Son `pre_seconds` + sonraki `post_seconds` saniyeyi bir klip olarak kaydeder."""
        if self._writer is not None:
            self._log("Zaten kayıt sürüyor, bekle.")
            return None
        if not self._buffer:
            self._log("Kayıt için kare yok.")
            return None

        label = "".join(c for c in label if c.isalnum() or c in "-_") or "unlabeled"
        folder = os.path.join(self.out_root, label)
        os.makedirs(folder, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(folder, f"{stamp}.mp4")

        h, w = self._buffer[-1].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
        if not writer.isOpened():
            self._log(f"Klip yazılamadı (codec?): {path}")
            return None

        for f in list(self._buffer):
            if f.shape[:2] == (h, w):
                writer.write(f)

        self._writer = writer
        self._writer_path = path
        self._writer_deadline = time.time() + self.post_seconds

        if note:
            with open(path[:-4] + ".txt", "w", encoding="utf-8") as fh:
                fh.write(note + "\n")

        self._log(f"Klip kaydı başladı: {label}/{stamp}.mp4 (+{self.post_seconds:.0f} sn)")
        return path

    def _finish(self):
        if self._writer is not None:
            self._writer.release()
            self._log(f"Klip kaydedildi: {os.path.relpath(self._writer_path, self.out_root)}")
            self._writer = None
            self._writer_path = None

    def close(self):
        self._finish()
