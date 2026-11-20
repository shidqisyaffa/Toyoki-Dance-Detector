from pathlib import Path
from typing import Union
import pygame


class AudioPlayer:
    def __init__(self, music_path: Union[str, Path]):
        self.music_path = Path(music_path)
        self._initialized = False
        self._available = False
        self._volume = 0.8
        self._muted = False

    def _init_mixer(self) -> bool:
        if self._initialized:
            return self._available

        if not self.music_path.exists():
            print(f"[AudioPlayer] File musik tidak ditemukan: {self.music_path}")
            self._initialized = True
            self._available = False
            return False

        try:
            pygame.mixer.init()
            pygame.mixer.music.load(str(self.music_path))
            pygame.mixer.music.set_volume(self._volume)
            self._initialized = True
            self._available = True
            return True
        except Exception as e:
            print(f"[AudioPlayer] Gagal menginisialisasi audio: {e}")
            self._initialized = True
            self._available = False
            return False

    def play_once(self) -> None:
        if not self._init_mixer():
            return
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play(0)

    def play_loop(self) -> None:
        if not self._init_mixer():
            return
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play(-1)

    def is_playing(self) -> bool:
        if not self._initialized or not self._available:
            return False
        return pygame.mixer.music.get_busy()

    def stop(self) -> None:
        if not self._initialized or not self._available:
            return
        pygame.mixer.music.stop()

    def set_volume(self, volume: float) -> float:
        self._volume = max(0.0, min(volume, 1.0))
        if self._available:
            pygame.mixer.music.set_volume(0.0 if self._muted else self._volume)
        return self._volume

    def change_volume(self, delta: float) -> float:
        return self.set_volume(self._volume + delta)

    def toggle_mute(self) -> bool:
        self._muted = not self._muted
        if self._available:
            pygame.mixer.music.set_volume(0.0 if self._muted else self._volume)
        return self._muted

    @property
    def volume(self) -> float:
        return self._volume

    @property
    def is_muted(self) -> bool:
        return self._muted

    def shutdown(self) -> None:
        if not self._initialized:
            return
        if self._available:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except Exception:
                pass
        self._initialized = False
        self._available = False
