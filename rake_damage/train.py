"""제진기 레이크 파손 감지 v2 - YOLOv8n 전이학습 (그룹 분할, 증강 누수 차단)
데이터 345장 (고유원본 115개), train 279 / valid 66"""
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,          # Hailo-8L 호환
        batch=16,
        device=0,
        patience=20,
        project="rake_damage_v2", name="exp1",
        # Roboflow에서 이미 증강된 데이터라 학습측 증강은 약하게
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.3,
        fliplr=0.5, mosaic=0.5, mixup=0.0,
    )
    m = model.val()
    print(f"mAP50: {m.box.map50:.3f}  mAP50-95: {m.box.map:.3f}")
