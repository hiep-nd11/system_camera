#!/usr/bin/env python3
import json
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from threading import Thread, Event
from typing import Dict, Optional
from dataclasses import dataclass
import signal

from core.services.object_detection import ObjectDetector
from core.services.object_tracking import TrackerFactory, TrackingAdapter
from core.features.rabbitmq_service import RabbitMQService
from core.features.suicide_detection import SuicideDetector, SuicideStatus
from core.gstreamer.gst_reader_simple import RTSPReaderSimple
from core.gstreamer.gst_writer import RTSPWriter
from core.utils.fps_pipeline import FPSCounter
from core.services.rabbitmq_client import RabbitMQClient
from core.utils.path_resolver import resolve_path

__all__ = ["SuicideDetectionPipeline", "SuicideDetectionSystem"]


class SuicideDetectionPipeline:

    
    def __init__(self, camera_id: str, config: dict, default_rtsp_port: int = 8554):
        self.camera_id = camera_id
        self.config = config
        self.running = False
        self.stop_event = Event()

        model_path_raw = config.get("model")
        if model_path_raw:
            self.model_path = str(resolve_path(model_path_raw))
        else:
            self.model_path = None
        
        self.rtsp_in = config.get("rtsp_in") or config.get("rtsp_link")
        
        self.rtsp_out = config.get("rtsp_out", "")
        
        rtsp_out_port_config = config.get("port_rtsp_out", default_rtsp_port)
        if isinstance(rtsp_out_port_config, str):
            self.rtsp_out_port = int(rtsp_out_port_config) if rtsp_out_port_config else default_rtsp_port
        else:
            self.rtsp_out_port = rtsp_out_port_config or default_rtsp_port
        
        if self.rtsp_out:
            self.rtsp_out_path = self.rtsp_out if self.rtsp_out.startswith('/') else f"/{self.rtsp_out}"
        else:
            self.rtsp_out_path = f"/{camera_id}"
        
        self.confidence_threshold = config.get("confidence_threshold", 0.25)
        self.iou_threshold = config.get("iou_threshold", 0.5)
        self.classes_to_count = config.get("classes_to_count", [0])
        
        self.danger_zones = config.get("danger_zones", [])
        
        self.track_thresh = config.get("track_thresh", 0.4)
        self.track_buffer = config.get("track_buffer", 120)
        self.match_thresh = config.get("match_thresh", 0.7)
        
        self.overlap_threshold = config.get("overlap_threshold", 0.5)
        self.warning_time = config.get("warning_time", 30.0)
        self.corner_cross_threshold = config.get("corner_cross_threshold", 3)
        
        self.queue_name = config.get("name_queue", "suicide-detection")
        queue_port = config.get("port_queue", "5672")
        self.queue_port = int(queue_port) if isinstance(queue_port, str) else queue_port
        self.enable_rabbitmq = config.get("enable_rabbitmq", True)
        
        if not self.model_path or not self.rtsp_in:
            raise ValueError(f"Config for {camera_id} missing required fields: model and rtsp_in/rtsp_link")
        
        if not self.danger_zones or len(self.danger_zones) == 0:
            raise ValueError(f"Config for {camera_id} must have at least 1 danger zone")
        
        self.reader = None
        self.detector = None
        self.tracker_adapter = None
        self.suicide_detector = None
        self.writer = None
        self.rabbitmq_client = None
        
        self.last_sent_alerts = set()
        
        self.fps_counter = FPSCounter(f"Pipeline_{camera_id}", window_size=30)
        self.total_frames = 0
        self.last_stats_time = time.time()
        self.stats_interval = 10.0
        
        print(f"Pipeline Initialized: {camera_id}")
        print(f"Model: {Path(self.model_path).name}")
        print(f"RTSP IN: {self.rtsp_in}")
        print(f"RTSP OUT: rtsp://localhost:{self.rtsp_out_port}{self.rtsp_out_path}")
        print(f"RTSP Port: {self.rtsp_out_port}")
        print(f"Confidence: {self.confidence_threshold}")
        print(f"IOU Threshold: {self.iou_threshold}")
        print(f"Classes to Detect: {self.classes_to_count}")
        print(f"Danger Zones: {len(self.danger_zones)} zones")
        print(f"Overlap Threshold: {self.overlap_threshold}")
        print(f"Warning Time: {self.warning_time}s")
        print(f"Corner Cross Threshold: {self.corner_cross_threshold}")

    def _initialize_rabbitmq_client(self) -> Optional[RabbitMQClient]:
        return RabbitMQService.initialize_client(
            default_port=self.queue_port,
            enabled=self.enable_rabbitmq
        )

    def _initialize_components(self):
        print(f"[{self.camera_id}] Initializing components...")
        
        self.reader = RTSPReaderSimple(
            rtsp_url=self.rtsp_in,
            latency=100,
            show_info=True
        )
        
        self.detector = ObjectDetector(
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            use_half=False,
            img_size=640
        )
        
        tracker = TrackerFactory.create_tracker(
            tracker_type="BYTE_TRACK",
            track_thresh=self.track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            frame_rate=30
        )
        self.tracker_adapter = TrackingAdapter(tracker)
        
        self.suicide_detector = SuicideDetector(
            danger_zones=self.danger_zones,
            overlap_threshold=self.overlap_threshold,
            warning_time=self.warning_time,
            corner_cross_threshold=self.corner_cross_threshold
        )
        
        self.reader.start()
        
        frame = None
        for attempt in range(150):
            frame = self.reader.get_latest_frame()
            if frame is not None:
                height, width = frame.shape[:2]
                print(f" Got first frame: {width}x{height} (after {attempt * 0.2:.1f}s)")
                break
            
            if (attempt + 1) % 25 == 0:
                print(f" Still waiting... ({(attempt + 1) * 0.2:.0f}s elapsed)")
            
            time.sleep(0.2)
        
        if frame is None:
            raise RuntimeError(
                f"Failed to get first frame from {self.rtsp_in}\n"
                f"   Please check:\n"
                f"   - Camera is online and accessible\n"
                f"   - RTSP URL is correct\n"
                f"   - Username/password are correct\n"
                f"   - Network connectivity\n"
                f"   You can test with: vlc {self.rtsp_in}"
            )
        
        self.writer = RTSPWriter(
            rtsp_path=self.rtsp_out_path,
            port=self.rtsp_out_port,
            width=width,
            height=height,
            fps=30,
            use_hw_encode=True,
            show_info=False
        )
        
        self.rabbitmq_client = self._initialize_rabbitmq_client()
        
        print(f"[{self.camera_id}] All components initialized\n")
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        detections = self.detector.detect(frame)
        
        online_targets = []
        if detections is not None:
            online_targets = self.tracker_adapter.update(detections, frame)
        
        filtered_targets = []
        for target in online_targets:
            x1, y1, x2, y2, tid, score, cls = target
            if cls in self.classes_to_count:
                filtered_targets.append([x1, y1, x2, y2, tid])
        
        if filtered_targets:
            targets_array = np.array(filtered_targets)
            self.suicide_detector.update(targets_array)
            
            if self.rabbitmq_client:
                current_alerts = set(self.suicide_detector.suicide_alerts)
                new_alerts = current_alerts - self.last_sent_alerts
                
                for track_id in new_alerts:
                    message = {
                        "camera_id": self.camera_id,
                        "alert_type": "suicide",
                        "track_id": int(track_id),
                        "status": "SUICIDE",
                        "timestamp": time.time()
                    }
                    if self.rabbitmq_client.publish(message):
                        self.last_sent_alerts.add(track_id)
        
        output_frame = frame.copy()
        
        if filtered_targets:
            targets_array = np.array(filtered_targets)
            output_frame = self.suicide_detector.draw_zones_and_status(output_frame, targets_array)
        else:
            output_frame = self.suicide_detector.draw_zones_only(output_frame)
        
        height, width = output_frame.shape[:2]
        
        box_width = 300
        box_height = 120
        
        left = width - box_width - 20
        top = height - box_height - 20
        right = width - 20
        bottom = height - 20
        
        cv2.rectangle(output_frame, (left, top), (right, bottom), (0, 0, 0), -1)
        cv2.rectangle(output_frame, (left, top), (right, bottom), (255, 255, 255), 2)
        
        text_x = left + 20
        fps = self.fps_counter.get_fps()
        
        cv2.putText(output_frame, f"Camera: {self.camera_id}", (text_x, top + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output_frame, f"FPS: {fps:.1f}", (text_x, top + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(output_frame, f"Objects: {len(filtered_targets)}", (text_x, top + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return output_frame
    
    def _run(self):
        
        try:
            while not self.stop_event.is_set():
                frame = self.reader.get_latest_frame()
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                try:
                    output_frame = self._process_frame(frame)
                    
                    self.writer.write_frame(output_frame)
                    
                    self.fps_counter.tick()
                    self.total_frames += 1
                    
                    current_time = time.time()
                    if current_time - self.last_stats_time >= self.stats_interval:
                        self._print_stats()
                        self.last_stats_time = current_time
                    
                except Exception as e:
                    print(f"[{self.camera_id}] Error processing frame: {e}")
                    continue
        
        except Exception as e:
            print(f"[{self.camera_id}] Fatal error in processing loop: {e}")
        
        finally:
            print(f"[{self.camera_id}] Processing stopped")
    
    def _print_stats(self):
        fps = self.fps_counter.get_fps()
        avg_fps = self.fps_counter.get_average_fps()
        suicide_alerts = len(self.suicide_detector.suicide_alerts)
        
        print(f"\n{'='*70}")
        print(f"[{self.camera_id}] Statistics")
        print(f"{'='*70}")
        print(f"   Total Frames: {self.total_frames}")
        print(f"   Current FPS: {fps:.2f}")
        print(f"   Average FPS: {avg_fps:.2f}")
        print(f"   Suicide Alerts: {suicide_alerts}")
        if suicide_alerts > 0:
            print(f"   Alert IDs: {self.suicide_detector.suicide_alerts}")
        print(f"   RTSP URL: rtsp://localhost:{self.rtsp_out_port}{self.rtsp_out_path}")
        print(f"{'='*70}\n")
    
    def start(self) -> bool:
        if self.running:
            print(f"[{self.camera_id}] Already running")
            return False
        
        try:
            self._initialize_components()
            
            if not self.writer.start():
                raise RuntimeError("Failed to start RTSP writer")
            
            self.running = True
            self.stop_event.clear()
            self.thread = Thread(target=self._run, daemon=False)
            self.thread.start()
            
            print(f"[{self.camera_id}] Pipeline started successfully\n")
            return True
            
        except Exception as e:
            print(f"[{self.camera_id}] Failed to start: {e}")
            self.stop()
            return False
    
    def stop(self):
        """Stop the pipeline"""
        if not self.running:
            return
        
        print(f"\n[{self.camera_id}] Stopping pipeline...")
        
        self.running = False
        self.stop_event.set()
        
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        if self.writer:
            self.writer.stop()
        if self.reader:
            self.reader.stop()
        if self.rabbitmq_client:
            self.rabbitmq_client.close()
        
        self._print_stats()
        print(f"[{self.camera_id}] Pipeline stopped\n")
    
    def is_running(self) -> bool:
        return self.running


class SuicideDetectionSystem:
    def __init__(self, config_file: str, base_rtsp_port: int = 8554, setup_signal_handlers: bool = True):
        self.config_file = config_file
        self.base_rtsp_port = base_rtsp_port
        self.pipelines: Dict[str, SuicideDetectionPipeline] = {}
        self.running = False
        
        self._load_config()
        
        if setup_signal_handlers:
            try:
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
            except ValueError as e:
                print(f"Cannot setup signal handlers (not in main thread): {e}")
    
    def _load_config(self):
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print(f"   Found {len(config)} camera(s)")
            for camera_id in config.keys():
                print(f"   - {camera_id}")
            print(f"{'='*70}\n")
            
            self.config = config
            
        except Exception as e:
            sys.exit(1)
    
    def _signal_handler(self, signum, frame):
        self.stop()
        sys.exit(0)
    
    def start(self):
        for idx, (camera_id, camera_config) in enumerate(self.config.items()):
            try:
                default_port = self.base_rtsp_port + idx
                
                pipeline = SuicideDetectionPipeline(
                    camera_id=camera_id,
                    config=camera_config,
                    default_rtsp_port=default_port
                )
                
                if pipeline.start():
                    self.pipelines[camera_id] = pipeline
                    print(f"{camera_id} started successfully")
                else:
                    print(f"Failed to start {camera_id}")
                
                time.sleep(2)
                
            except Exception as e:
                print(f"Error starting {camera_id}: {e}")
                import traceback
                traceback.print_exc()
        
        if not self.pipelines:
            print("No pipelines started successfully")
            return False
        
        self.running = True
        
        print(f"   Active pipelines: {len(self.pipelines)}")
        for camera_id, pipeline in self.pipelines.items():
            rtsp_url = f"rtsp://localhost:{pipeline.rtsp_out_port}{pipeline.rtsp_out_path}"
            print(f"   - {camera_id}: {rtsp_url}")
        print(f"\nView streams using VLC or ffplay:")
        for camera_id, pipeline in self.pipelines.items():
            rtsp_url = f"rtsp://localhost:{pipeline.rtsp_out_port}{pipeline.rtsp_out_path}"
            print(f"   vlc {rtsp_url}")
        print(f"{'='*70}\n")
        
        return True
    
    def run(self):
        if not self.start():
            print("Failed to start system")
            return
                
        try:
            while self.running:
                dead_pipelines = [
                    camera_id for camera_id, pipeline in self.pipelines.items()
                    if not pipeline.is_running()
                ]
                
                if dead_pipelines:
                    print(f"Pipelines stopped: {dead_pipelines}")
                
                time.sleep(5)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.stop()
    
    def stop(self):
        if not self.running:
            return
        self.running = False
        
        for camera_id, pipeline in self.pipelines.items():
            try:
                pipeline.stop()
            except Exception as e:
                print(f"Error stopping {camera_id}: {e}")
        

def main():
    
    config_file = "core/configs/cf_suicide_detection.json"
    
    if not Path(config_file).exists():
        print(f"Config file not found: {config_file}")
        sys.exit(1)
    
    system = SuicideDetectionSystem(
        config_file=config_file,
        base_rtsp_port=8561 
    )
    
    system.run()


if __name__ == "__main__":
    main()

