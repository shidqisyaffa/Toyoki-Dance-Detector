from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import math
import time

import cv2
import numpy as np
from PIL import Image, ImageSequence


def find_asset_file(
    assets_dir: Union[str, Path], extensions: List[str], preferred_name: str = ""
) -> Optional[Path]:
    """Mencari file asset secara otomatis berdasarkan ekstensi."""
    folder = Path(assets_dir)
    if not folder.exists():
        return None

    # Cek preferred name dulu
    if preferred_name:
        candidate = folder / preferred_name
        if candidate.exists():
            return candidate

    # Cari file yang cocok dengan salah satu ekstensi
    for ext in extensions:
        matches = list(folder.glob(f"*{ext}"))
        if matches:
            valid = [m for m in matches if not m.name.startswith(".")]
            if valid:
                return valid[0]

    return None


class GifPlayer:
    def __init__(self, gif_path: Union[str, Path], target_width: int = 240) -> None:
        self.gif_path = Path(gif_path)
        self.target_width = target_width
        self.frames: List[np.ndarray] = []
        self.durations: List[float] = []
        self.total_duration: float = 0.0
        self.playing: bool = False
        self.start_time: float = 0.0
        self.loaded: bool = False
        self._load()

    def _load(self) -> None:
        if not self.gif_path.exists():
            self.loaded = False
            return

        try:
            img = Image.open(str(self.gif_path))
        except Exception as e:
            print(f"[GifPlayer] Gagal membuka GIF: {e}")
            self.loaded = False
            return

        for frame in ImageSequence.Iterator(img):
            duration_ms = frame.info.get("duration", 80)
            if duration_ms <= 0:
                duration_ms = 80

            # Muat sebagai RGBA untuk mendukung transparansi
            frame_rgba = frame.convert("RGBA")
            frame_np = np.array(frame_rgba)

            # Resize frame dengan mempertahankan aspect ratio
            h, w = frame_np.shape[:2]
            scale = self.target_width / max(w, 1)
            target_h = int(h * scale)
            resized = cv2.resize(
                frame_np, (self.target_width, target_h), interpolation=cv2.INTER_AREA
            )

            # Ubah dari RGBA (PIL) ke BGRA (OpenCV)
            b = resized[:, :, 2]
            g = resized[:, :, 1]
            r = resized[:, :, 0]
            a = resized[:, :, 3]
            frame_bgra = np.dstack([b, g, r, a])

            self.frames.append(frame_bgra)
            self.durations.append(duration_ms / 1000.0)

        self.total_duration = sum(self.durations) if self.durations else 0.0
        self.loaded = bool(self.frames)

    def start(self, now: float) -> None:
        if not self.loaded:
            return
        self.playing = True
        self.start_time = now

    def stop(self) -> None:
        self.playing = False

    def get_frame(self, now: float) -> Optional[np.ndarray]:
        if not self.playing or not self.loaded or self.total_duration <= 0:
            return None
        elapsed = (now - self.start_time) % self.total_duration
        acc = 0.0
        for frame, duration in zip(self.frames, self.durations):
            acc += duration
            if elapsed <= acc:
                return frame
        return self.frames[-1]


class GifWindow:
    """Window opsional jika pengguna ingin pop-up terpisah."""

    def __init__(self, title: str = "Dancing Cat") -> None:
        self.title = title
        self.is_open = False

    def show(self, frame_bgra: np.ndarray) -> None:
        if frame_bgra is None:
            return
        if frame_bgra.shape[2] == 4:
            bgr = frame_bgra[:, :, :3]
        else:
            bgr = frame_bgra
        cv2.imshow(self.title, bgr)
        self.is_open = True

    def hide(self) -> None:
        if self.is_open:
            cv2.destroyWindow(self.title)
            self.is_open = False


def overlay_image_alpha(
    background: np.ndarray,
    overlay_bgra: np.ndarray,
    pos: Tuple[int, int],
) -> None:
    """Menempelkan gambar RGBA/BGRA ke background BGR dengan alpha blending."""
    x, y = pos
    bh, bw = background.shape[:2]
    oh, ow = overlay_bgra.shape[:2]

    if x >= bw or y >= bh or x + ow <= 0 or y + oh <= 0:
        return

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bw, x + ow), min(bh, y + oh)

    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    overlay_crop = overlay_bgra[oy1:oy2, ox1:ox2]
    bg_crop = background[y1:y2, x1:x2]

    if overlay_crop.shape[2] == 4:
        alpha = (overlay_crop[:, :, 3] / 255.0)[:, :, np.newaxis]
        overlay_color = overlay_crop[:, :, :3]
        bg_crop[:] = (alpha * overlay_color + (1.0 - alpha) * bg_crop).astype(np.uint8)
    else:
        bg_crop[:] = overlay_crop


def draw_hud(
    frame: np.ndarray,
    data: Dict,
    fps: float = 0.0,
    dance_live: bool = False,
    cat_frame: Optional[np.ndarray] = None,
    volume: float = 0.8,
    is_muted: bool = False,
    toast_message: str = "",
) -> None:
    """Menggambar HUD modern, status, indikator deteksi, efek party, dan overlay GIF PIP."""
    h, w = frame.shape[:2]

    # Efek Party Border Berdenyut saat Tarian Aktif
    if dance_live:
        pulse = int(127 + 127 * math.sin(time.time() * 8))
        neon_color = (0, pulse, 255)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), neon_color, 4)

    # 1. Gambar Area Mulut & Jarak Ambang
    if data.get("mouth_center"):
        mx, my = data["mouth_center"]
        threshold = int(data.get("mouth_threshold", 70))
        is_covered = data.get("is_mouth_covered", False)

        mouth_color = (0, 255, 0) if is_covered else (0, 220, 255)
        cv2.circle(frame, (int(mx), int(my)), 5, mouth_color, -1)
        cv2.circle(frame, (int(mx), int(my)), threshold, mouth_color, 1, cv2.LINE_AA)

    # 2. Gambar Tangan (Mouth Hand & Wave Hand)
    mouth_hand = data.get("mouth_hand")
    wave_hand = data.get("wave_hand")

    for hand in data.get("hands", []):
        is_mouth_h = hand is mouth_hand
        is_wave_h = hand is wave_hand

        if is_mouth_h:
            color = (0, 255, 120)
            tag = "Tutup Mulut"
        elif is_wave_h:
            color = (0, 200, 255)
            tag = "Melambai"
        else:
            color = (180, 180, 180)
            tag = "Tangan"

        for x, y in hand["landmarks"]:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)

        px, py = hand["palm_center"]
        cv2.circle(frame, (int(px), int(py)), 6, color, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            tag,
            (int(px) - 25, int(py) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    # 3. Lintasan Ayunan Tangan (Wave Trajectory)
    trajectory = data.get("trajectory", [])
    if len(trajectory) > 1:
        for i in range(1, len(trajectory)):
            p0 = trajectory[i - 1]
            p1 = trajectory[i]
            alpha = i / len(trajectory)
            color = (int(50 * (1 - alpha)), int(200 * alpha), int(255))
            thickness = max(1, int(3 * alpha))
            cv2.line(
                frame,
                (int(p0[0]), int(p0[1])),
                (int(p1[0]), int(p1[1])),
                color,
                thickness,
                cv2.LINE_AA,
            )

    # 4. Header Badge Status (Modern HUD Pill)
    state = data.get("state", "IDLE")
    wave = data.get("wave")

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (390, 115), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    if dance_live:
        state_title = "DANCE PARTY ACTIVE! 🎵"
        state_color = (0, 255, 255)
        guide_text = "Goyang terus! Asikkk Toyoki Time! 🐱"
    elif state == "MOUTH_COVERED":
        state_title = "MULUT TERTUTUP"
        state_color = (0, 255, 120)
        guide_text = "Tahan posisi mulut, siapkan tangan satunya..."
    elif state == "READY_TO_WAVE" or state == "WAITING_HAND":
        state_title = "SIAP MELAMBAI"
        state_color = (0, 220, 255)
        guide_text = "Lambaikan tangan satunya ke kiri & kanan!"
    elif state == "KEEP_WAVING" or state == "WAVING_DETECTED":
        state_title = "MELAMBAI..."
        state_color = (255, 180, 0)
        guide_text = "Lambaikan lagi sedikit untuk mulai pesta!"
    elif state == "DANCE_DETECTED" or state == "ACTIVE_DANCE":
        state_title = "DANCE TRIGGERED!"
        state_color = (0, 255, 0)
        guide_text = "ASIKK! Goyang terus!"
    else:
        state_title = "STANDBY"
        state_color = (200, 200, 200)
        guide_text = "Tutup mulutmu dengan 1 tangan untuk mulai"

    cv2.putText(
        frame,
        state_title,
        (22, 38),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        state_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        guide_text,
        (22, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    # Bar Confidence
    conf = wave.confidence if wave else 0.0
    bar_x, bar_y, bar_w, bar_h = 22, 78, 200, 10
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_w, bar_y + bar_h),
        (80, 80, 80),
        -1,
    )
    fill_w = int(bar_w * conf)
    fill_color = (0, 255, 0) if conf >= 0.70 else (0, 220, 255)
    if fill_w > 0:
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + fill_w, bar_y + bar_h),
            fill_color,
            -1,
        )

    conf_text = f"Wave: {int(conf * 100)}%"
    cv2.putText(
        frame,
        conf_text,
        (bar_x + bar_w + 12, bar_y + 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # 5. Picture-in-Picture (PIP) Animasi GIF Kucing
    if dance_live and cat_frame is not None:
        cat_h, cat_w = cat_frame.shape[:2]
        pip_x = w - cat_w - 20
        pip_y = 20

        bg_card = frame.copy()
        cv2.rectangle(
            bg_card,
            (pip_x - 6, pip_y - 6),
            (pip_x + cat_w + 6, pip_y + cat_h + 6),
            (25, 25, 25),
            -1,
        )
        cv2.addWeighted(bg_card, 0.55, frame, 0.45, 0, frame)

        overlay_image_alpha(frame, cat_frame, (pip_x, pip_y))

        border_color = (0, 255, 255) if (int(time.time() * 4) % 2 == 0) else (0, 200, 255)
        cv2.rectangle(
            frame,
            (pip_x - 6, pip_y - 6),
            (pip_x + cat_w + 6, pip_y + cat_h + 6),
            border_color,
            2,
            cv2.LINE_AA,
        )

    # 6. Toast Notification (jika ada pesan popup sementara)
    if toast_message:
        tw = len(toast_message) * 9 + 40
        tx = (w - tw) // 2
        ty = h - 55
        cv2.rectangle(frame, (tx, ty), (tx + tw, ty + 30), (30, 30, 30), -1)
        cv2.rectangle(frame, (tx, ty), (tx + tw, ty + 30), (0, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            toast_message,
            (tx + 18, ty + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 120),
            1,
            cv2.LINE_AA,
        )

    # 7. Info FPS, Volume & Tombol Pintasan di Footer
    vol_str = "MUTED" if is_muted else f"{int(volume * 100)}%"
    footer_text = f"FPS: {fps:.1f} | Vol: {vol_str} | [q]: Keluar | [s]: Foto | [+/-]: Vol | [m]: Mute"
    cv2.putText(
        frame,
        footer_text,
        (15, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
