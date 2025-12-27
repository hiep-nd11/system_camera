
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from enum import Enum

import cv2
import numpy as np

__all__ = ["SuicideDetector", "SuicideStatus"]


class SuicideStatus(Enum):
    """Trạng thái theo dõi của đối tượng"""
    NORMAL = "normal"
    WARNING = "warning"      
    SUSPICIOUS = "suspicious" 
    SUICIDE = "suicide"      


class SuicideDetector:
    def __init__(
        self,
        danger_zones: List[Dict[str, Any]], 
        overlap_threshold: float = 0.5,
        warning_time: float = 30.0,
        corner_cross_threshold: int = 3
    ) -> None:
        self.danger_zones = []
        for zone_config in danger_zones:
            zone_polygon = np.array(zone_config["zone"], dtype=np.int32)
            danger_line = zone_config["danger_line"] 
            self.danger_zones.append({
                "polygon": zone_polygon,
                "danger_line": danger_line
            })
            
        self.overlap_threshold = overlap_threshold
        self.warning_time = warning_time  
        self.corner_cross_threshold = corner_cross_threshold

        self._track_states: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {
                "status": SuicideStatus.NORMAL,
                "zone_id": -1,          
                "enter_time": 0.0,      
                "last_seen": 0.0        
            }
        )
        
        self.suicide_alerts = []  
    
    @staticmethod
    def _calculate_overlap_ratio(bbox: Tuple[int, int, int, int], 
                               polygon: np.ndarray) -> float:
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        if bbox_area <= 0:
            return 0.0
            
        mask = np.zeros((y2 - y1 + 100, x2 - x1 + 100), dtype=np.uint8)
        
        shifted_polygon = polygon.copy()
        shifted_polygon[:, 0] -= (x1 - 50)  
        shifted_polygon[:, 1] -= (y1 - 50)
        
        cv2.fillPoly(mask, [shifted_polygon], 255)
        
        bbox_in_mask = (50, 50, x2 - x1 + 50, y2 - y1 + 50)
        bbox_mask = np.zeros_like(mask)
        cv2.rectangle(bbox_mask, 
                     (bbox_in_mask[0], bbox_in_mask[1]),
                     (bbox_in_mask[2], bbox_in_mask[3]), 
                     255, -1)
        
        intersection = cv2.bitwise_and(mask, bbox_mask)
        intersection_area = np.sum(intersection > 0)
        
        return intersection_area / bbox_area if bbox_area > 0 else 0.0

    @staticmethod  
    def _get_bbox_corners(bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        x1, y1, x2, y2 = bbox
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        
    @staticmethod
    def _point_crosses_line(point: Tuple[int, int], 
                          line: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
        x, y = point
        (x1, y1), (x2, y2) = line
        
        A = y2 - y1
        B = x1 - x2  
        C = x2 * y1 - x1 * y2
        
        distance = abs(A * x + B * y + C) / np.sqrt(A * A + B * B) if (A * A + B * B) > 0 else float('inf')
        return distance <= 10 
        
    def _count_corners_cross_line(self, bbox: Tuple[int, int, int, int],
                                line: Tuple[Tuple[int, int], Tuple[int, int]]) -> int:
        corners = self._get_bbox_corners(bbox)
        cross_count = 0
        for corner in corners:
            if self._point_crosses_line(corner, line):
                cross_count += 1
        return cross_count
    
    def update(self, online_targets: np.ndarray) -> None:
        """Cập nhật detector với danh sách targets"""
        now = time.time()
        active_track_ids = set()
        
        for t in online_targets:
            x1, y1, x2, y2, track_id = map(int, t[:5])
            active_track_ids.add(track_id)
            bbox = (x1, y1, x2, y2)
            
            state = self._track_states[track_id]
            state["last_seen"] = now
            
            max_overlap = 0.0
            best_zone_id = -1
            
            for zone_id, zone_data in enumerate(self.danger_zones):
                overlap = self._calculate_overlap_ratio(bbox, zone_data["polygon"])
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_zone_id = zone_id
                    
            current_status = state["status"]
            
            if max_overlap >= self.overlap_threshold:
                if current_status == SuicideStatus.NORMAL:
                    state["status"] = SuicideStatus.WARNING
                    state["zone_id"] = best_zone_id
                    state["enter_time"] = now
                    
                elif current_status == SuicideStatus.WARNING:
                    zone_data = self.danger_zones[state["zone_id"]]
                    corners_crossed = self._count_corners_cross_line(bbox, zone_data["danger_line"])
                    
                    if corners_crossed >= self.corner_cross_threshold:
                        state["status"] = SuicideStatus.SUICIDE
                        if track_id not in self.suicide_alerts:
                            self.suicide_alerts.append(track_id)
                    else:
                        if now - state["enter_time"] >= self.warning_time:
                            state["status"] = SuicideStatus.SUSPICIOUS
                        
                elif current_status == SuicideStatus.SUSPICIOUS:
                    zone_data = self.danger_zones[state["zone_id"]]
                    corners_crossed = self._count_corners_cross_line(bbox, zone_data["danger_line"])
                    
                    if corners_crossed >= self.corner_cross_threshold:
                        state["status"] = SuicideStatus.SUICIDE
                        if track_id not in self.suicide_alerts:
                            self.suicide_alerts.append(track_id)
                            
            else:
                if current_status in [SuicideStatus.WARNING, SuicideStatus.SUSPICIOUS]:
                    state["status"] = SuicideStatus.NORMAL
                    state["zone_id"] = -1
                    state["enter_time"] = 0.0
                    
        expire_time = now - 10.0
        expired_tracks = [tid for tid, state in self._track_states.items() 
                         if state["last_seen"] < expire_time]
        for tid in expired_tracks:
            del self._track_states[tid]
            
    def get_track_status(self, track_id: int) -> SuicideStatus:
        return self._track_states[track_id]["status"]
        
    def get_warning_time_remaining(self, track_id: int) -> float:
        state = self._track_states[track_id]
        if state["status"] != SuicideStatus.WARNING:
            return 0.0
        elapsed = time.time() - state["enter_time"]
        return max(0.0, self.warning_time - elapsed)
        
    def draw_zones_only(self, frame: np.ndarray) -> np.ndarray:
        
        for zone_data in self.danger_zones:
            cv2.polylines(frame, [zone_data["polygon"]], True, (0, 255, 255), 2)
            cv2.line(frame, zone_data["danger_line"][0], zone_data["danger_line"][1], 
                    (0, 0, 255), 3)
        
        alert_text = f"Suicide Alerts: {len(self.suicide_alerts)}"
        cv2.putText(frame, alert_text, (10, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                   
        return frame

    def draw_zones_and_status(self, frame: np.ndarray, 
                            online_targets: np.ndarray) -> np.ndarray:
        """Vẽ các vùng nguy hiểm và trạng thái lên frame"""
        
        frame = self.draw_zones_only(frame)
                    
        if len(online_targets) > 0:
            for t in online_targets:
                x1, y1, x2, y2, track_id = map(int, t[:5])
                status = self.get_track_status(track_id)
                
                if status == SuicideStatus.NORMAL:
                    color = (0, 255, 0)      
                    text = f"ID:{track_id}"
                elif status == SuicideStatus.WARNING:
                    color = (0, 165, 255)    
                    remaining = self.get_warning_time_remaining(track_id)
                    text = f"ID:{track_id} WARNING {remaining:.1f}s"
                elif status == SuicideStatus.SUSPICIOUS:
                    color = (0, 0, 255)      
                    text = f"ID:{track_id} SUSPICIOUS"
                else:  
                    color = (255, 0, 255)    
                    text = f"ID:{track_id} ⚠️ SUICIDE"
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                   
        return frame
