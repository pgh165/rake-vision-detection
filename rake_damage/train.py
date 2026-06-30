"""제진기 레이크 파손 감지 - YOLOv8n 전이학습 (누수 방지 split)"""
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data="data.yaml",
        epochs=50,          # val 곡선상 10~15에서 정점 → 50이면 충분
        imgsz=640,          # Hailo-8L 호환
        batch=16,
        device=0,
        patience=10,        # 조기종료 빠르게
        project="rake_damage", name="exp_clean",
        # 데이터 적을 때 증강 약하게
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.2,
        fliplr=0.5, mosaic=0.3, mixup=0.0,
    )
    m = model.val()
    print(f"mAP50: {m.box.map50:.3f}  mAP50-95: {m.box.map:.3f}")
