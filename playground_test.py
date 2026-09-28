"""
Çocuk Oyun Alanı - Sade OpenCV Test Uygulaması (yedek/hafif sürüm)
====================================================================

NOT: Ana test uygulaması `desktop_app.py` (modern Qt arayüzü).
Bu dosya Qt kurmadan hızlıca denemek için sade OpenCV penceresi sürümü.
Mantığın tamamı `tracking_core.SafetyMonitor` içindedir.

NASIL ÇALIŞTIRILIR?
    pip install -r requirements.txt
    python playground_test.py

TUŞLAR
  q : çıkış
  r : hedef seçimini sıfırla
  f : kavga tespitini aç/kapat
  a : AI kavga doğrulamasını aç/kapat (model yüklenir)
  c : "kavga" klibi kaydet   (dataset/kavga/)
  n : "normal" klibi kaydet  (dataset/normal/)
"""

import cv2

from tracking_core import SafetyMonitor, KP_MIN_CONF
from clip_recorder import ClipRecorder

CAMERA_INDEX = 0

SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class PlaygroundTestApp:
    def __init__(self):
        self.monitor = SafetyMonitor()
        self.monitor.on_event = lambda msg: print(f"[OLAY] {msg}")
        self.recorder = ClipRecorder(fps=15.0)
        self.recorder.on_event = lambda msg: print(f"[KLIP] {msg}")
        self.click_point = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_point = (x, y)

    def _draw_skeleton(self, frame, keypoints):
        for a, b in SKELETON_EDGES:
            if a < len(keypoints) and b < len(keypoints):
                xa, ya, ca = keypoints[a]
                xb, yb, cb = keypoints[b]
                if ca >= KP_MIN_CONF and cb >= KP_MIN_CONF:
                    cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (255, 200, 0), 2)
        for x, y, c in keypoints:
            if c >= KP_MIN_CONF:
                cv2.circle(frame, (int(x), int(y)), 3, (255, 200, 0), -1)

    def _draw(self, frame, status):
        target_id = status["target_id"]
        tracks = self.monitor.tracker.tracks
        fighters = set()
        for a, b in status.get("fighting_pairs", []):
            fighters.add(a)
            fighters.add(b)

        for track_id, track in tracks.items():
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            is_target = (track_id == target_id)
            if track_id in fighters:
                color = (255, 0, 200)
            elif is_target:
                color = (0, 0, 255)
            else:
                color = (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                          3 if (is_target or track_id in fighters) else 2)

            label = f"HEDEF #{track_id}" if is_target else f"#{track_id}"
            if is_target and status["fall_state"] == "monitoring":
                label += f"  YERDE {int(status['seconds_on_ground'])}sn"
            elif is_target and status["fall_state"] == "alarm":
                label += "  DUSME!"
            if track_id in fighters:
                label += "  KAVGA?"
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if is_target and track.keypoints is not None:
                self._draw_skeleton(frame, track.keypoints)

        for a, b in status.get("fighting_pairs", []):
            if a in tracks and b in tracks:
                ca = ((tracks[a].bbox[0] + tracks[a].bbox[2]) / 2, (tracks[a].bbox[1] + tracks[a].bbox[3]) / 2)
                cb = ((tracks[b].bbox[0] + tracks[b].bbox[2]) / 2, (tracks[b].bbox[1] + tracks[b].bbox[3]) / 2)
                cv2.line(frame, (int(ca[0]), int(ca[1])), (int(cb[0]), int(cb[1])), (255, 0, 200), 3)

        status_text, status_color = self._status_line(status)
        cv2.putText(frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        fight_hint = "acik" if status.get("fight_enabled") else "KAPALI"
        cv2.putText(frame, f"q: cikis | r: hedefi sifirla | f: kavga tespiti ({fight_hint})",
                    (15, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if status["alarm_active"]:
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 8)

    @staticmethod
    def _status_line(status):
        if status["alarm_active"] and status["alarm_reason"] == "fall":
            return f"DUSME ALARMI! ({int(status['seconds_on_ground'])} sn yerde)", (0, 0, 255)
        if status["alarm_active"] and status["alarm_reason"] == "fight":
            return "OLASI KAVGA (dusuk guven)", (255, 0, 200)
        if status["alarm_active"] and status["alarm_reason"] == "missing":
            return f"KAYIP ALARMI! ({(status['missing_elapsed'] or 0):.0f} sn)", (0, 0, 255)
        if status["fall_state"] == "monitoring":
            remaining = max(0.0, status["ground_alarm_seconds"] - status["seconds_on_ground"])
            return f"Hedef yerde - alarm {remaining:.0f} sn sonra", (0, 165, 255)
        if status["target_state"] == "missing":
            remaining = max(0.0, status["missing_alarm_seconds"] - (status["missing_elapsed"] or 0))
            return f"Kayip - alarm {remaining:.1f} sn sonra", (0, 165, 255)
        if status["target_state"] == "present":
            return "Takip ediliyor", (0, 200, 0)
        return "Hedef secilmedi - bir kutuya tikla", (0, 255, 255)

    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError("Kamera açılamadı. CAMERA_INDEX değerini kontrol et.")

        window_name = "Cocuk Oyun Alani - Test"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        print("Model yukleniyor...")
        self.monitor.load_model()
        print("Hazir. Kendi kutuna tikla; sonra kadraj disina cik ya da yere uzan.")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Kameradan kare okunamadi, cikiliyor.")
                    break

                if self.click_point is not None:
                    x, y = self.click_point
                    self.click_point = None
                    if not self.monitor.select_target_at(x, y):
                        print("[OLAY] Tiklanan noktada kimse yok.")

                self.recorder.add(frame)  # HAM kare (cizimden once)
                status = self.monitor.process(frame)
                self._draw(frame, status)

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.monitor.reset_target()
                elif key == ord('f'):
                    self.monitor.set_fight_enabled(not self.monitor.fight.enabled)
                    print(f"[OLAY] Kavga tespiti: {'acik' if self.monitor.fight.enabled else 'kapali'}")
                elif key == ord('a'):
                    on = not (self.monitor.fight.use_ai and self.monitor.fight.scorer is not None)
                    print(f"[OLAY] AI kavga dogrulamasi: {'aciliyor...' if on else 'kapaniyor'}")
                    self.monitor.set_violence_ai_enabled(on)
                elif key == ord('c'):
                    self.recorder.save("kavga", note="manuel")
                elif key == ord('n'):
                    self.recorder.save("normal", note="manuel")

        finally:
            self.recorder.close()
            self.monitor.shutdown()
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    PlaygroundTestApp().run()
