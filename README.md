# Toyoki Dance Detector 🐱

Aplikasi deteksi gerakan interaktif berbasis AI (**MediaPipe Vision & OpenCV**) yang mendeteksi tren tarian *Kicau Mania / Toyoki Dance* secara real-time melalui webcam.

---

## 🚀 Fitur Utama

- **Auto-Detect Asset:** Otomatis mendeteksi file lagu (`.mp3`, `.wav`) dan animasi (`.gif`) apa pun di dalam folder `assets/`.
- **Picture-in-Picture (PIP) Overlay:** Kucing menari langsung muncul di pojok video kamera dengan efek transparansi (*alpha blending*), tanpa perlu membuka jendela terpisah yang mengganggu.
- **Natural Mirror Mode:** Mode cermin webcam sehingga gerakan terasa alami dan teks HUD tetap terbaca tegak lurus.
- **Fleksibel Kiri/Kanan:** Bisa menggunakan tangan kiri atau kanan untuk menutup mulut, sementara tangan satunya lagi melambai.
- **Dinamis Terhadap Resolusi:** Jarak jangkauan mulut ke tangan dinormalisasi secara otomatis berdasarkan ukuran wajah.
- **HUD Interaktif:** Menampilkan status gerakan, bar kekuatan lambaian tangan (*wave confidence*), jejak gerakan tangan (*trajectory*), dan FPS.

---

## 📁 Struktur File

```text
projek3/
├── assets/
│   ├── models/                           # Model MediaPipe (.task)
│   ├── Cute_Toyoki_Cat_Dance_Template.gif # Animasi kucing menari
│   └── Lirik lagu toyoki.mp3             # Lagu Toyoki
├── audio.py                              # Pengelola audio background (pygame-ce)
├── detector.py                           # Detektor MediaPipe (Hand + Face Mesh)
├── motion.py                             # Analisis osilasi & gelombang lambaian tangan
├── ui.py                                 # Pengolah GIF, PIP overlay & HUD display
├── main.py                               # Titik masuk utama program
└── requirements.txt                      # Daftar dependensi Python
```

---

## 🎮 Cara Menjalankan

1. Pastikan webcam terhubung ke komputer.
2. Jalankan perintah berikut di PowerShell / Terminal:
   ```powershell
   .\projek3\Scripts\python.exe main.py
   ```

---

## 🕺 Cara Bermain

1. **Arahkan wajah ke webcam** dalam posisi tegak dan pencahayaan cukup.
2. **Tutup mulut** dengan telapak tangan salah satu tangan (kiri atau kanan).
3. **Lambaikan tangan yang satunya lagi** secara horizontal ke kiri dan ke kanan.
4. **Party Time!** Lagu akan otomatis berputar dan kucing lucu akan menari di layar!
5. Untuk berhenti, cukup turunkan tangan. Tekan tombol **`q`** atau **`ESC`** di keyboard untuk keluar.

---

## ⚙️ Pengaturan Tambahan (`main.py`)

Di bagian atas file `main.py`, kamu bisa mengubah:
- `CAMERA_INDEX`: Ubah ke `1` jika ingin menggunakan webcam USB eksternal.
- `MIRROR_PREVIEW`: Set `True` untuk mode cermin (bawaan: `True`).
- `DANCE_GRACE_S`: Batas toleransi jeda gerakan sebelum musik berhenti (bawaan: `1.2` detik).
- `SEPARATE_GIF_WINDOW`: Set `True` jika ingin kucing tampil di jendela terpisah selain PIP.

