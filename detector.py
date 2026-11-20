from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

from motion import WaveMotionTracker, WaveResult

MOUTH_LANDMARKS = [13, 14, 78, 308, 61, 291]
MOUTH_CORNER_LEFT = 61
MOUTH_CORNER_RIGHT = 291

MODEL_DIR = Path("assets/models")
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
FACE_MODEL_PATH = MODEL_DIR / "face_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/face_landmarker.task"


@dataclass
class DetectorConfig:
    # Jarak relatif tangan ke mulut (faktor skala lebar mulut)
    mouth_scale_factor: float = 2.4
    fallback_mouth_distance_px: float = 95.0
    wave_speed_px_s: float = 75.0         # Kecepatan lambaian responsif
    wave_dir_changes: int = 2             # Cukup 2 kali balik arah kiri-kanan
    wave_min_move_px: float = 8.0         # Sensitif terhadap lambaian cepat
    wave_window_s: float = 0.75           # Window analisis dipersingkat agar cepat berhenti saat idle
    wave_horizontal_ratio: float = 0.80   # Toleran terhadap ayunan melengkung/miring
    mouth_hold_s: float = 0.12            # Respon tutup mulut cepat
    wave_hold_s: float = 0.14             # Respon picu cepat
    cooldown_s: float = 0.5               # Cooldown singkat (0.5s)


class KicauDetector:
    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()
        self._ensure_models()

        base_options = mp.tasks.BaseOptions
        vision = mp.tasks.vision
        running_mode = vision.RunningMode.VIDEO

        hand_options = vision.HandLandmarkerOptions(
            base_options=base_options(model_asset_path=str(HAND_MODEL_PATH)),
            running_mode=running_mode,
            num_hands=2,
            min_hand_detection_confidence=0.50,
            min_hand_presence_confidence=0.50,
            min_tracking_confidence=0.50,
        )
        face_options = vision.FaceLandmarkerOptions(
            base_options=base_options(model_asset_path=str(FACE_MODEL_PATH)),
            running_mode=running_mode,
            num_faces=1,
            min_face_detection_confidence=0.50,
            min_face_presence_confidence=0.50,
            min_tracking_confidence=0.50,
        )

        self.hands = vision.HandLandmarker.create_from_options(hand_options)
        self.face_mesh = vision.FaceLandmarker.create_from_options(face_options)
        self.motion = WaveMotionTracker(
            window_seconds=self.config.wave_window_s,
            min_speed=self.config.wave_speed_px_s,
            min_dir_changes=self.config.wave_dir_changes,
            min_move_px=self.config.wave_min_move_px,
            min_horizontal_ratio=self.config.wave_horizontal_ratio,
        )

        self.state = "IDLE"
        self.mouth_start: Optional[float] = None
        self.wave_start: Optional[float] = None
        self.cooldown_until: float = 0.0
        self._last_timestamp_ms: int = -1

    def process(self, frame_bgr: np.ndarray, timestamp: float) -> Dict:
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Pastikan timestamp strictly monotonically increasing untuk MediaPipe
        timestamp_ms = int(timestamp * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        hands_result = self.hands.detect_for_video(image, timestamp_ms)
        face_result = self.face_mesh.detect_for_video(image, timestamp_ms)

        mouth_center, mouth_threshold = self._get_mouth_data(face_result, width, height)

        detected_hands: List[Dict] = []
        if hands_result.hand_landmarks and hands_result.handedness:
            for landmarks, handedness in zip(
                hands_result.hand_landmarks, hands_result.handedness
            ):
                label = handedness[0].category_name.upper()
                score = handedness[0].score
                hand_data = self._extract_hand(landmarks, label, score, width, height)
                detected_hands.append(hand_data)

        mouth_hand = None
        waving_hand = None

        if mouth_center and detected_hands:
            # Urutkan berdasarkan jarak terdekat mulut ke jari/telapak tangan
            hands_sorted = sorted(
                detected_hands,
                key=lambda h: self._hand_mouth_distance(mouth_center, h),
            )
            closest_hand = hands_sorted[0]
            dist_to_mouth = self._hand_mouth_distance(mouth_center, closest_hand)

            if dist_to_mouth <= mouth_threshold:
                mouth_hand = closest_hand
                if len(hands_sorted) > 1:
                    waving_hand = hands_sorted[1]
            else:
                if len(detected_hands) > 0:
                    waving_hand = detected_hands[0]
        elif detected_hands:
            waving_hand = detected_hands[0]

        is_mouth_covered = mouth_hand is not None

        wave_result = WaveResult(False, 0.0, 0, 0.0, 0.0)
        if waving_hand:
            # Lacak pusat telapak tangan (palm_center) yang memiliki amplitude ayunan jauh lebih lebar dari pergelangan (wrist)
            hx, hy = waving_hand["palm_center"]
            wave_result = self.motion.update(hx, hy, timestamp)
        else:
            # JANGAN reset buffer jika hanya 1 frame drop saat lambaian cepat!
            wave_result = self.motion.decay_or_prune(timestamp)

        has_waving_hand = waving_hand is not None
        triggered, display_state = self._update_state(
            is_mouth_covered, has_waving_hand, wave_result, timestamp
        )

        return {
            "mouth_center": mouth_center,
            "mouth_threshold": mouth_threshold,
            "hands": detected_hands,
            "mouth_hand": mouth_hand,
            "wave_hand": waving_hand,
            "is_mouth_covered": is_mouth_covered,
            "wave": wave_result,
            "state": display_state,
            "triggered": triggered,
        }

    def reset_state(self) -> None:
        """Mereset state detector ke IDLE secara instan tanpa menunggu cooldown."""
        self.state = "IDLE"
        self.mouth_start = None
        self.wave_start = None
        self.cooldown_until = 0.0
        self.motion.reset()

    def shutdown(self) -> None:
        try:
            self.hands.close()
            self.face_mesh.close()
        except Exception:
            pass

    def _update_state(
        self,
        mouth_covered: bool,
        wave_hand_present: bool,
        wave: WaveResult,
        now: float,
    ) -> Tuple[bool, str]:
        triggered = False
        display_state = self.state

        if self.state == "IDLE":
            if mouth_covered:
                self.state = "MOUTH_COVERED"
                self.mouth_start = now
                display_state = self.state

        elif self.state == "MOUTH_COVERED":
            if not mouth_covered:
                self.state = "IDLE"
                self.mouth_start = None
                display_state = "IDLE"
            elif self.mouth_start and (now - self.mouth_start >= self.config.mouth_hold_s):
                self.state = "WAVING"
                self.wave_start = None
                display_state = "READY_TO_WAVE"

        elif self.state == "WAVING":
            if not mouth_covered:
                self.state = "IDLE"
                self.wave_start = None
                display_state = "IDLE"
            else:
                if not wave_hand_present:
                    display_state = "WAITING_HAND"
                    self.wave_start = None
                    return triggered, display_state

                if wave.is_waving:
                    if self.wave_start is None:
                        self.wave_start = now
                        display_state = "WAVING_DETECTED"
                    elif (now - self.wave_start) >= self.config.wave_hold_s:
                        triggered = True
                        display_state = "DANCE_DETECTED"
                        self.state = "COOLDOWN"
                        self.cooldown_until = now + self.config.cooldown_s
                        self.wave_start = None
                else:
                    display_state = "KEEP_WAVING"
                    self.wave_start = None

        elif self.state == "COOLDOWN":
            display_state = "ACTIVE_DANCE"
            if now >= self.cooldown_until:
                self.state = "IDLE"
                self.mouth_start = None
                self.wave_start = None

        return triggered, display_state

    def _hand_mouth_distance(
        self, mouth_center: Tuple[float, float], hand: Dict
    ) -> float:
        """Menghitung jarak terdekat dari mulut ke jari-jemari atau telapak tangan."""
        key_indices = [0, 4, 8, 9, 12, 16, 20]
        points = [hand["landmarks"][i] for i in key_indices] + [hand["palm_center"]]
        return min(self._distance(mouth_center, pt) for pt in points)

    def _get_mouth_data(
        self, face_result, width: int, height: int
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        if not face_result.face_landmarks:
            return None, self.config.fallback_mouth_distance_px

        face_landmarks = face_result.face_landmarks[0]
        xs = [face_landmarks[i].x * width for i in MOUTH_LANDMARKS]
        ys = [face_landmarks[i].y * height for i in MOUTH_LANDMARKS]
        mouth_center = (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))

        p_left = face_landmarks[MOUTH_CORNER_LEFT]
        p_right = face_landmarks[MOUTH_CORNER_RIGHT]
        mouth_width = math.hypot(
            (p_left.x - p_right.x) * width,
            (p_left.y - p_right.y) * height,
        )

        mouth_threshold = max(
            mouth_width * self.config.mouth_scale_factor,
            height * 0.15,  # Minimal 15% tinggi frame
        )

        return mouth_center, float(mouth_threshold)

    def _extract_hand(
        self, landmarks, label: str, score: float, width: int, height: int
    ) -> Dict:
        pts = [(lm.x * width, lm.y * height) for lm in landmarks]
        palm_ids = [0, 5, 9, 13, 17]
        palm_x = sum(pts[i][0] for i in palm_ids) / len(palm_ids)
        palm_y = sum(pts[i][1] for i in palm_ids) / len(palm_ids)
        return {
            "label": label,
            "confidence": score,
            "landmarks": pts,
            "wrist": pts[0],
            "palm_center": (palm_x, palm_y),
        }

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _ensure_models(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._download_if_missing(HAND_MODEL_PATH, HAND_MODEL_URL)
        self._download_if_missing(FACE_MODEL_PATH, FACE_MODEL_URL)

    @staticmethod
    def _download_if_missing(path: Path, url: str) -> None:
        if path.exists() and path.stat().st_size > 1000:
            return
        tmp_path = path.with_suffix(".tmp")
        print(f"[Detector] Mengunduh model AI: {path.name} (hanya sekali)...")
        try:
            urllib.request.urlretrieve(url, str(tmp_path))
            if tmp_path.exists():
                if path.exists():
                    path.unlink()
                tmp_path.rename(path)
                print(f"[Detector] Berhasil mengunduh: {path.name}")
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(
                f"Gagal mengunduh model {path.name} dari {url}. Pastikan ada koneksi internet: {e}"
            )
