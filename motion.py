from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple


@dataclass
class WaveResult:
    is_waving: bool
    confidence: float
    direction_changes: int
    avg_speed: float
    avg_horizontal_ratio: float


class WaveMotionTracker:
    def __init__(
        self,
        window_seconds: float = 0.75,
        min_speed: float = 75.0,
        min_dir_changes: int = 2,
        min_move_px: float = 8.0,
        min_horizontal_ratio: float = 0.80,
    ) -> None:
        self.window_seconds = window_seconds
        self.min_speed = min_speed
        self.min_dir_changes = min_dir_changes
        self.min_move_px = min_move_px
        self.min_horizontal_ratio = min_horizontal_ratio
        self._points: Deque[Tuple[float, float, float]] = deque()
        self.last_result = WaveResult(False, 0.0, 0, 0.0, 0.0)

    def reset(self) -> None:
        self._points.clear()
        self.last_result = WaveResult(False, 0.0, 0, 0.0, 0.0)

    def update(self, x: float, y: float, timestamp: float) -> WaveResult:
        self._points.append((timestamp, x, y))
        self._prune(timestamp)

        if len(self._points) < 4:
            self.last_result = WaveResult(False, 0.0, 0, 0.0, 0.0)
            return self.last_result

        dir_changes, avg_speed, avg_ratio = self._analyze()
        confidence = self._score(dir_changes, avg_speed, avg_ratio)

        is_waving = (
            avg_speed >= self.min_speed
            and dir_changes >= self.min_dir_changes
            and avg_ratio >= self.min_horizontal_ratio
        )

        self.last_result = WaveResult(
            is_waving=is_waving,
            confidence=confidence,
            direction_changes=dir_changes,
            avg_speed=avg_speed,
            avg_horizontal_ratio=avg_ratio,
        )
        return self.last_result

    def decay_or_prune(self, timestamp: float) -> WaveResult:
        """Mempertahankan state lambaian saat terjadi frame drop sesaat (motion blur)."""
        self._prune(timestamp)
        if len(self._points) < 3:
            self.last_result = WaveResult(False, 0.0, 0, 0.0, 0.0)
            return self.last_result

        dir_changes, avg_speed, avg_ratio = self._analyze()
        confidence = self._score(dir_changes, avg_speed, avg_ratio) * 0.90
        is_waving = (
            avg_speed >= self.min_speed
            and dir_changes >= self.min_dir_changes
            and avg_ratio >= self.min_horizontal_ratio
        )
        self.last_result = WaveResult(
            is_waving=is_waving,
            confidence=confidence,
            direction_changes=dir_changes,
            avg_speed=avg_speed,
            avg_horizontal_ratio=avg_ratio,
        )
        return self.last_result

    def get_points(self) -> List[Tuple[float, float]]:
        return [(p[1], p[2]) for p in self._points]

    def _prune(self, now: float) -> None:
        while self._points and (now - self._points[0][0]) > self.window_seconds:
            self._points.popleft()

    def _analyze(self) -> Tuple[int, float, float]:
        if len(self._points) < 2:
            return 0, 0.0, 0.0

        t_start = self._points[0][0]
        t_end = self._points[-1][0]
        total_time = max(t_end - t_start, 0.01)

        total_x_dist = 0.0
        total_y_dist = 0.0

        for i in range(1, len(self._points)):
            _, x0, y0 = self._points[i - 1]
            _, x1, y1 = self._points[i]
            total_x_dist += abs(x1 - x0)
            total_y_dist += abs(y1 - y0)

        avg_speed = total_x_dist / total_time
        avg_ratio = total_x_dist / max(total_y_dist, 1.0)

        # Analisis osilasi kiri-kanan dengan hysteresis adaptif
        dir_changes = 0
        current_dir = 0  # 1: ke kanan, -1: ke kiri
        last_extreme_x = self._points[0][1]

        for i in range(1, len(self._points)):
            x = self._points[i][1]
            diff = x - last_extreme_x

            if current_dir >= 0:
                if diff > 0:
                    last_extreme_x = x
                elif -diff >= self.min_move_px:
                    if current_dir == 1:
                        dir_changes += 1
                    current_dir = -1
                    last_extreme_x = x
            else:
                if diff < 0:
                    last_extreme_x = x
                elif diff >= self.min_move_px:
                    dir_changes += 1
                    current_dir = 1
                    last_extreme_x = x

        return dir_changes, avg_speed, avg_ratio

    def _score(self, dir_changes: int, avg_speed: float, avg_ratio: float) -> float:
        speed_score = min(avg_speed / max(self.min_speed, 1.0), 1.5)
        change_score = min(dir_changes / max(self.min_dir_changes, 1), 1.5)
        ratio_score = min(avg_ratio / max(self.min_horizontal_ratio, 0.1), 1.5)
        score = (speed_score * 0.40 + change_score * 0.45 + ratio_score * 0.15) / 1.5
        return float(max(0.0, min(score, 1.0)))
