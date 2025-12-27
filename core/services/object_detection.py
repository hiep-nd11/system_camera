import torch
from ultralytics import YOLO
import numpy as np

class ObjectDetector:
    def __init__(self, model_path, confidence_threshold=0.25, iou_threshold=0.5, 
                 use_half=True, img_size=640):

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            self.use_half = use_half and self.device != "cpu"
            if self.use_half:
                self.model.model.half()  

            dummy = torch.zeros((1, 3, self.img_size, self.img_size)).to(self.device)
            if self.use_half:
                dummy = dummy.half()
            for _ in range(3):
                _ = self.model.predict(dummy, verbose=False)
                        
        except Exception as e:
            raise IOError(f"Không thể tải model: {model_path}. Lỗi: {e}")
    
    def detect(self, frame):

        try:
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                half=self.use_half,  
                verbose=False,
                imgsz=self.img_size
            )
            
            if results and results[0].boxes:
                return results[0].boxes
            return None
            
        except Exception as e:
            print(f"Detection error: {e}")
            return None
    
    def detect_batch(self, frames, batch_size=8):
        try:
            all_results = []
            
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i + batch_size]
                
                results = self.model.predict(
                    batch,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    half=self.use_half,
                    verbose=False,
                    imgsz=self.img_size,
                    stream=True 
                )
                
                for result in results:
                    if result.boxes:
                        all_results.append(result.boxes)
                    else:
                        all_results.append(None)
            
            return all_results
            
        except Exception as e:
            print(f"Batch detection error: {e}")
            return [None] * len(frames)
    
    def get_gpu_memory_usage(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            return f"VRAM: {allocated:.2f}GB / {reserved:.2f}GB reserved"
        return "CPU mode"