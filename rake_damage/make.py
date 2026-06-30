"""
학습된 모델로 영상에 파손 감지 박스를 입혀 결과 영상(mp4) 생성
- VideoWriter 열림 검증 + 코덱 폴백 추가 (0바이트 출력 방지)
"""
import cv2
import os
from ultralytics import YOLO

# ===== 설정 (본인 경로로 수정) =====
MODEL_PATH = "C:/Users/parkj/Desktop/capstone/rake_detection/rake_dataset_ready/runs/detect/rake_damage/exp_clean/weights/best.pt"   # ← Get-ChildItem으로 찾은 실제 경로
VIDEO_IN   = "C:/Users/parkj/Desktop/capstone/rake_detection/KakaoTalk_20260524_143357990.mp4"
VIDEO_OUT  = "result_detection.mp4"
CONF       = 0.15
# ==================================

# 입력 파일 존재 확인
assert os.path.exists(MODEL_PATH), f"모델 파일 없음: {MODEL_PATH}"
assert os.path.exists(VIDEO_IN),   f"영상 파일 없음: {VIDEO_IN}"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise RuntimeError(f"영상을 열 수 없음: {VIDEO_IN}")

fps   = cap.get(cv2.CAP_PROP_FPS) or 24.0
w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"입력 영상: {w}x{h} @ {fps:.1f}fps, {total}프레임")

# 코덱 폴백: mp4v -> avc1 -> XVID(avi) 순으로 시도
def open_writer(path, fps, size):
    for codec, ext in [("mp4v", ".mp4"), ("avc1", ".mp4"), ("XVID", ".avi")]:
        p = os.path.splitext(path)[0] + ext
        fourcc = cv2.VideoWriter_fourcc(*codec)
        wr = cv2.VideoWriter(p, fourcc, fps, size)
        if wr.isOpened():
            print(f"VideoWriter OK: codec={codec}, out={p}")
            return wr, p
        wr.release()
    raise RuntimeError("어떤 코덱으로도 VideoWriter를 열 수 없음")

out, out_path = open_writer(VIDEO_OUT, fps, (w, h))

written = 0
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, conf=CONF, device=0, verbose=False)
    annotated = results[0].plot()

    n = len(results[0].boxes)
    cv2.putText(annotated, f"damage: {n}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    out.write(annotated)
    written += 1
    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"{frame_idx}/{total} 처리 중... (이번 프레임 감지 {n}개)")

cap.release()
out.release()

# 결과 검증
size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
print(f"\n완료! {written}프레임 기록, 파일크기 {size/1e6:.2f}MB -> {out_path}")
if size == 0:
    print("⚠️ 파일이 0바이트! 아래 '추가 해결책'을 참고하세요.")