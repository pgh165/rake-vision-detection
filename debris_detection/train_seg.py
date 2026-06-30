"""제진기 부유물 세그멘테이션 - YOLOv8n-seg 전이학습 (누수 방지 split)"""
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n-seg.pt")   # 세그멘테이션 사전학습 가중치
    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,          # Hailo-8L 호환
        batch=16,
        device=0,
        patience=20,
        project="debris_seg", name="exp1",
        # 데이터 다양성이 파손보다 좋아서(80장) 증강은 표준 수준
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.4,
        fliplr=0.5, mosaic=0.7, mixup=0.0,
    )
    m = model.val()
    print(f"[Box]  mAP50: {m.box.map50:.3f}  mAP50-95: {m.box.map:.3f}")
    print(f"[Mask] mAP50: {m.seg.map50:.3f}  mAP50-95: {m.seg.map:.3f}")
