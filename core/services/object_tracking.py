# -*- coding: utf-8 -*-
import supervision as sv
import numpy as np
import logging
import torch

logger = logging.getLogger(__name__)

class TrackerFactory:
    @staticmethod
    def create_tracker(tracker_type="BYTE_TRACK", track_thresh=0.5, track_buffer=50, 
                      match_thresh=0.8, frame_rate=30, embedder_gpu=True):

        tracker_type_upper = tracker_type.upper()
        
        if tracker_type_upper in ["BYTE_TRACK", "BYTETRACK"]:
            return sv.ByteTrack(
                track_activation_threshold=track_thresh,
                lost_track_buffer=track_buffer,
                minimum_matching_threshold=match_thresh,
                frame_rate=frame_rate
            )
            
        else:
            raise ValueError(f"{tracker_type}")

class TrackingAdapter:
    def __init__(self, tracker_instance):
        self.tracker = tracker_instance
        self.is_supervision = isinstance(tracker_instance, sv.ByteTrack)
        
        self.frame_count = 0
        self.total_time = 0
    
    def update(self, detections, frame):

        if detections is None or frame is None:
            return []
            
        try:
            if hasattr(detections, 'data'):
                if len(detections.data) == 0:
                    return []
            elif isinstance(detections, np.ndarray):
                if detections.size == 0:
                    return []
            elif isinstance(detections, sv.Detections):
                if len(detections) == 0:
                    return []
        except:
            return []
        
        try:
            if self.is_supervision:
                return self._update_supervision(detections, frame)
            else:
                return self._update_deepsort(detections, frame)
        except Exception as e:
            logger.error(f"Tracking error: {e}")
            return []
    
    def _update_supervision(self, detections, frame):
        try:
            if isinstance(detections, sv.Detections):
                sv_detections = detections
            elif hasattr(detections, 'xyxy'):  
                sv_detections = sv.Detections(
                    xyxy=detections.xyxy.cpu().numpy(),
                    confidence=detections.conf.cpu().numpy(),
                    class_id=detections.cls.cpu().numpy().astype(int)
                )
            elif isinstance(detections, np.ndarray):
                if len(detections) == 0:
                    return []
                sv_detections = sv.Detections(
                    xyxy=detections[:, :4],
                    confidence=detections[:, 4] if detections.shape[1] > 4 else np.ones(len(detections)),
                    class_id=detections[:, 5].astype(int) if detections.shape[1] > 5 else np.zeros(len(detections), dtype=int)
                )
            else:
                return []
            
            tracked = self.tracker.update_with_detections(sv_detections)
            
            online_targets = []
            if tracked.tracker_id is not None:
                for i, tid in enumerate(tracked.tracker_id):
                    x1, y1, x2, y2 = tracked.xyxy[i]
                    score = tracked.confidence[i]
                    cls = tracked.class_id[i]
                    online_targets.append([
                        float(x1), float(y1), float(x2), float(y2),
                        int(tid), float(score), int(cls)
                    ])
            
            return online_targets
            
        except Exception as e:
            logger.error(f"ByteTrack error: {e}")
            return []
    
    def _update_deepsort(self, detections, frame):
        if self.DeepSort is None or not isinstance(self.tracker, self.DeepSort):
            return []
        
        try:
            ds_detections = []
            
            if hasattr(detections, 'xyxy'): 
                for i in range(len(detections)):
                    xyxy = detections.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    score = float(detections.conf[i])
                    cls = int(detections.cls[i])
                    ds_detections.append(([x1, y1, x2-x1, y2-y1], score, cls))
            
            elif isinstance(detections, np.ndarray):
                for det in detections:
                    x1, y1, x2, y2 = det[:4]
                    score = float(det[4]) if len(det) > 4 else 0.5
                    cls = int(det[5]) if len(det) > 5 else 0
                    ds_detections.append(([x1, y1, x2-x1, y2-y1], score, cls))
            
            elif isinstance(detections, sv.Detections):
                for i in range(len(detections)):
                    x1, y1, x2, y2 = detections.xyxy[i]
                    score = float(detections.confidence[i])
                    cls = int(detections.class_id[i])
                    ds_detections.append(([x1, y1, x2-x1, y2-y1], score, cls))
            else:
                return []
            
            if not ds_detections:
                return []
            

            tracks = self.tracker.update_tracks(ds_detections, frame=frame)
            
            online_targets = []
            for t in tracks:
                if not t.is_confirmed():
                    continue
                
                ltrb = t.to_ltrb()
                x1, y1, x2, y2 = ltrb
                tid = t.track_id
                score = t.det_conf if hasattr(t, 'det_conf') else 1.0
                cls = t.det_class if hasattr(t, 'det_class') else 0
                
                online_targets.append([
                    float(x1), float(y1), float(x2), float(y2),
                    int(tid), float(score), int(cls)
                ])
            
            return online_targets
            
        except Exception as e:
            logger.error(f"DeepSORT error: {e}")
            return []
    
    def get_fps(self):
        if self.frame_count > 0 and self.total_time > 0:
            return self.frame_count / self.total_time
        return 0.0
