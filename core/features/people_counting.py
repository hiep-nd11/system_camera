
import time
from collections import defaultdict
from typing import Dict, Tuple

import cv2
import numpy as np

__all__ = ["PeopleCounter"]


class PeopleCounter:  
    def __init__(
        self,
        zone: list[Tuple[int, int]],
        *,
        cooldown_sec: float = 2.0,
        state_expire_sec: float = 5.0,
    ) -> None:
        if not isinstance(zone, list) or len(zone) != 4:
            raise ValueError("4 điểm (x, y)")

        self.zone = np.array(zone, dtype=np.int32)

        mid1 = ((zone[0][0] + zone[3][0]) // 2, (zone[0][1] + zone[3][1]) // 2)
        mid2 = ((zone[1][0] + zone[2][0]) // 2, (zone[1][1] + zone[2][1]) // 2)
        self.counting_line: Tuple[Tuple[int, int], Tuple[int, int]] = (mid1, mid2)
        
        self._inside_sign: int = (
            np.sign(self._get_point_side(tuple(zone[0]), self.counting_line)) or 1
        )

        self.cooldown_sec = cooldown_sec
        self.state_expire_sec = state_expire_sec

        self._states: Dict[int, Dict[str, float | int]] = defaultdict(
            lambda: {"last_side": None, "last_seen": 0.0, "last_count_time": 0.0}
        )

        self.in_count = 0
        self.out_count = 0
        
    @staticmethod
    def _get_point_side(
        point: Tuple[int, int],
        line: Tuple[Tuple[int, int], Tuple[int, int]],
    ) -> int:
        x, y = point
        (x1, y1), (x2, y2) = line
        val = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        return int(np.sign(val))

    def _inside_zone(self, point: Tuple[int, int]) -> bool:
        return cv2.pointPolygonTest(self.zone, point, False) >= 0

    def update(self, online_targets: np.ndarray) -> None:
        now = time.time()
        active_ids = set()

        for t in online_targets:
            x1, y1, x2, y2, tid = map(int, t[:5])
            active_ids.add(tid)
            bottom_center = ((x1 + x2) // 2, y2)

            if not self._inside_zone(bottom_center):
                continue

            current_side = self._get_point_side(bottom_center, self.counting_line)
            if current_side == 0:
                continue  

            st = self._states[tid]
            last_side = st["last_side"]
            last_count = st["last_count_time"]

            if (
                last_side is not None
                and last_side != current_side
                and now - last_count >= self.cooldown_sec
            ):
                if current_side == self._inside_sign:
                    self.out_count += 1
                else:
                    self.in_count += 1
                st["last_count_time"] = now

            st["last_side"] = current_side
            st["last_seen"] = now

        expire_before = now - self.state_expire_sec
        for tid in [tid for tid, st in self._states.items() if st["last_seen"] < expire_before]:
            del self._states[tid]

    def reset_counters(self) -> None:
        self.in_count = 0
        self.out_count = 0
        self._states.clear()
        print(f"🔄 Đã reset bộ đếm về 0: IN={self.in_count}, OUT={self.out_count}")

    def get_counts(self) -> Tuple[int, int]:
        """Trả về (in_count, out_count)"""
        return (self.in_count, self.out_count)

    def draw_counters(self, frame: np.ndarray) -> np.ndarray:  
        cv2.polylines(frame, [self.zone], True, (0, 255, 255), 2)
        cv2.line(frame, *self.counting_line, (255, 0, 0), 3)
        # cv2.putText(frame, f"IN: {self.in_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        # cv2.putText(frame, f"OUT: {self.out_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        return frame
