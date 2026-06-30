"""
학습된 모델 → ONNX export (Hailo DFC 입력용)
Hailo-8L 호환: opset=11, dynamic=False, imgsz=640
"""
from ultralytics import YOLO

if __name__ == "__main__":
    # 학습 결과 best.pt 경로 (본인 학습 결과에 맞게 수정)
    model = YOLO("rake_damage/exp1/weights/best.pt")

    model.export(
        format="onnx",
        opset=11,          # Hailo DFC 호환
        dynamic=False,     # 고정 입력 크기 (Hailo 필수)
        imgsz=640,
        simplify=True,     # onnx-simplifier 적용
    )
    print("ONNX export 완료 → rake_damage/exp1/weights/best.onnx")
    print("다음: Hailo DFC로 int8 calibration → .hef 변환")
