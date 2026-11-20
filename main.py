from pathlib import Path
import time

import cv2

from audio import AudioPlayer
from detector import DetectorConfig, KicauDetector
from ui import GifPlayer, GifWindow, draw_hud, find_asset_file

ASSETS_DIR = Path("assets")

# Pengaturan
CAMERA_INDEX = 0       # Ubah ke 1 atau 2 jika menggunakan webcam eksternal (USB)
MIRROR_PREVIEW = True  # Mode cermin (sangat disarankan untuk webcam)
DANCE_GRACE_S = 0.65   # Jeda berhenti cepat (0.65s setelah lambaian berhenti)
MIN_DANCE_DURATION_S = 0.4  # Menghindari kedip sesaat di awal
SEPARATE_GIF_WINDOW = False  # Set True jika ingin window terpisah selain overlay PIP


def main() -> None:
    print("=" * 60)
    print("      KICAU MANIA / TOYOKI DANCE DETECTOR")
    print("=" * 60)

    # 1. Auto-detect Assets di folder assets
    music_file = find_asset_file(
        ASSETS_DIR, [".mp3", ".wav", ".ogg"], preferred_name="music.mp3"
    )
    gif_file = find_asset_file(
        ASSETS_DIR, [".gif"], preferred_name="cat.gif"
    )

    if music_file:
        print(f"[Assets] Musik terdeteksi : {music_file.name}")
    else:
        print("[Assets] Peringatan: Tidak ditemukan file musik (.mp3/.wav) di folder assets!")

    if gif_file:
        print(f"[Assets] GIF terdeteksi   : {gif_file.name}")
    else:
        print("[Assets] Peringatan: Tidak ditemukan file .gif di folder assets!")

    # 2. Inisialisasi Komponen dengan konfigurasi responsif
    config = DetectorConfig(
        mouth_scale_factor=2.4,
        fallback_mouth_distance_px=95.0,
        wave_speed_px_s=75.0,
        wave_dir_changes=2,
        wave_min_move_px=8.0,
        wave_window_s=0.75,
        wave_horizontal_ratio=0.80,
        mouth_hold_s=0.12,
        wave_hold_s=0.14,
        cooldown_s=0.5,
    )

    print("\n[AI] Menginisialisasi detektor MediaPipe...")
    detector = KicauDetector(config)
    audio = AudioPlayer(music_file) if music_file else None
    gif = GifPlayer(gif_file, target_width=220) if gif_file else None
    gif_window = GifWindow() if SEPARATE_GIF_WINDOW else None

    dance_live = False
    last_dance_active = 0.0
    dance_start_time = 0.0

    toast_msg = ""
    toast_until = 0.0

    print(f"[Camera] Membuka webcam (index {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[Error] Gagal membuka webcam. Pastikan kamera terpasang dan tidak dipakai aplikasi lain.")
        detector.shutdown()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n[Siap] Arahkan wajah ke webcam:")
    print(" 1. Tutup mulutmu dengan satu tangan.")
    print(" 2. Lambaikan tangan satunya ke kiri & kanan.")
    print(" 3. Goyang terus! Begitu kamu berhenti melambai, musik & kucing langsung berhenti.\n")

    prev_time = time.time()
    fps = 30.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[Camera] Gagal membaca frame dari webcam.")
                break

            now = time.time()
            dt = max(now - prev_time, 1e-4)
            prev_time = now
            current_fps = 1.0 / dt
            fps = 0.9 * fps + 0.1 * current_fps

            # Flip horizontal frame sejak awal agar seperti cermin
            if MIRROR_PREVIEW:
                frame = cv2.flip(frame, 1)

            # Proses deteksi AI
            data = detector.process(frame, now)
            data["trajectory"] = detector.motion.get_points()

            # Cek status gerakan
            wave = data.get("wave")
            is_waving = bool(wave and wave.is_waving)
            is_mouth_covered = bool(data.get("is_mouth_covered"))

            # Tarian dianggap aktif hanya jika tangan sedang MELAMBAI aktif
            if dance_live:
                dance_active = is_waving and (
                    is_mouth_covered or (wave and wave.confidence >= 0.40)
                )
            else:
                dance_active = is_mouth_covered and is_waving

            if dance_active:
                last_dance_active = now

            # Jika terpicu (Trigger)
            if data.get("triggered", False):
                dance_live = True
                last_dance_active = now
                dance_start_time = now
                if audio:
                    audio.play_loop()
                if gif:
                    gif.start(now)

            # Jika gerakan berhenti:
            if dance_live and not dance_active:
                # Jika kedua tangan benar-benar turun dari kamera, berhenti instan (0.35s)
                hands_count = len(data.get("hands", []))
                grace_limit = 0.35 if hands_count == 0 else DANCE_GRACE_S

                if (now - dance_start_time) >= MIN_DANCE_DURATION_S:
                    if (now - last_dance_active) > grace_limit:
                        dance_live = False
                        detector.reset_state()
                        if audio:
                            audio.stop()
                        if gif:
                            gif.stop()
                        if gif_window:
                            gif_window.hide()

            # Dapatkan frame GIF jika aktif
            cat_frame = None
            if dance_live and gif:
                cat_frame = gif.get_frame(now)
                if gif_window and cat_frame is not None:
                    gif_window.show(cat_frame)

            # Toast notifikasi
            current_toast = toast_msg if now < toast_until else ""

            # Gambar HUD modern & Picture-in-Picture GIF
            draw_hud(
                frame,
                data,
                fps=fps,
                dance_live=dance_live,
                cat_frame=cat_frame,
                volume=audio.volume if audio else 0.8,
                is_muted=audio.is_muted if audio else False,
                toast_message=current_toast,
            )

            cv2.imshow("Kicau Mania - Toyoki Dance Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            elif key in (ord("s"), ord("S")):
                # Tangkap dan simpan foto / screenshot
                screenshots_dir = Path("screenshots")
                screenshots_dir.mkdir(exist_ok=True)
                filename = f"toyoki_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                save_path = screenshots_dir / filename
                cv2.imwrite(str(save_path), frame)
                toast_msg = f"Foto Disimpan: {filename}"
                toast_until = now + 2.5
            elif key in (ord("+"), ord("=")):
                if audio:
                    v = audio.change_volume(0.1)
                    toast_msg = f"Volume: {int(v * 100)}%"
                    toast_until = now + 1.5
            elif key in (ord("-"), ord("_")):
                if audio:
                    v = audio.change_volume(-0.1)
                    toast_msg = f"Volume: {int(v * 100)}%"
                    toast_until = now + 1.5
            elif key in (ord("m"), ord("M")):
                if audio:
                    muted = audio.toggle_mute()
                    toast_msg = "Muted" if muted else f"Volume: {int(audio.volume * 100)}%"
                    toast_until = now + 1.5

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if gif_window:
            gif_window.hide()
        detector.shutdown()
        if audio:
            audio.shutdown()
        print("\n[Selesai] Program ditutup dengan aman. Sampai jumpa!")


if __name__ == "__main__":
    main()
