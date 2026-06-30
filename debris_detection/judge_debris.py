"""
부유물 양 판단 (덩어리 통합 방식)
- 개별 마스크를 형태학 연산(dilate+close)으로 하나의 부유물 영역으로 통합
- 통합된 영역의 면적 비율로 많다/적다 판단
- 시간 평활화(최근 N프레임 평균)로 숫자 안정화
- 개별 라벨 제거, 부유물 영역을 반투명 오버레이로 표시
"""
import cv2
import numpy as np
import os
from collections import deque
from ultralytics import YOLO

# ===== 설정 =====
MODEL_PATH = "C:/Users/parkj/Desktop/capstone/rake_detection/debris/runs/segment/debris_seg/exp1/weights/best.pt"
VIDEO_IN   = "C:/Users/parkj/Desktop/capstone/rake_detection/KakaoTalk_20260620_121546018.mp4"
VIDEO_OUT  = "result_debris_cluster.mp4"

CONF       = 0.15    # 마스크 감지 임계값
DILATE     = 35      # 덩어리 통합 강도 (클수록 멀리 있는 것도 이어짐, 20~50에서 조절)
SMOOTH_N   = 15      # 시간 평활화: 최근 N프레임 평균
TH_LOW     = 0.08    # 이하 = 적음
TH_HIGH    = 0.25    # 이상 = 많음
# ===============

assert os.path.exists(MODEL_PATH), f"모델 없음: {MODEL_PATH}"
assert os.path.exists(VIDEO_IN),   f"영상 없음: {VIDEO_IN}"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_IN)
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def open_writer(path, fps, size):
    for codec, ext in [("mp4v",".mp4"),("avc1",".mp4"),("XVID",".avi")]:
        p = os.path.splitext(path)[0]+ext
        wr = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*codec), fps, size)
        if wr.isOpened():
            print(f"VideoWriter OK: {codec} -> {p}"); return wr, p
        wr.release()
    raise RuntimeError("VideoWriter 실패")

out, out_path = open_writer(VIDEO_OUT, fps, (W, H))

# 형태학 커널 (한 번만 생성)
k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE, DILATE))
k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE*2, DILATE*2))

ratio_hist = deque(maxlen=SMOOTH_N)   # 시간 평활화용 버퍼
idx = 0
while True:
    ret, frame = cap.read()
    if not ret: break

    results = model.predict(frame, conf=CONF, device=0, verbose=False, retina_masks=True)
    masks = results[0].masks

    # 1) 모든 마스크를 원본 해상도로 합치기
    union = np.zeros((H, W), np.uint8)
    n_raw = 0
    if masks is not None and len(masks.data) > 0:
        n_raw = len(masks.data)
        m = masks.data.any(dim=0).cpu().numpy().astype(np.uint8)
        # 모델 출력 마스크가 프레임 크기와 다르면 리사이즈
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        union = m

    # 2) dilate + close 로 흩어진 마스크를 덩어리로 통합
    merged = cv2.dilate(union, k_dilate, iterations=1)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, k_close)

    # 3) 통합 덩어리 개수 & 면적
    n_cluster, _ = cv2.connectedComponents(merged)
    n_cluster -= 1   # 배경 제외
    raw_ratio = float(merged.sum()) / (H * W)

    # 4) 시간 평활화
    ratio_hist.append(raw_ratio)
    ratio = sum(ratio_hist) / len(ratio_hist)

    # 5) 단계 판단
    if ratio < TH_LOW:      level, color = "ADEQUATE (low)", (0,200,0)
    elif ratio < TH_HIGH:   level, color = "MODERATE",       (0,200,255)
    else:                   level, color = "HIGH - CLEAN!",  (0,0,255)

    # 6) 시각화: 통합 덩어리를 반투명 색으로 오버레이
    overlay = frame.copy()
    overlay[merged > 0] = color
    annotated = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
    # 덩어리 외곽선
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours, -1, color, 3)

    # 7) 정보 패널
    cv2.rectangle(annotated, (10,10), (560,120), (0,0,0), -1)
    cv2.putText(annotated, f"debris: {ratio*100:.1f}%", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
    cv2.putText(annotated, level, (20,95),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(annotated, f"clusters:{n_cluster} (raw obj:{n_raw})", (300,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    out.write(annotated)
    idx += 1
    if idx % 30 == 0:
        print(f"{idx}/{total} ... {ratio*100:.1f}% {level} clusters={n_cluster}")

cap.release(); out.release()
size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
print(f"\n완료! {idx}프레임, {size/1e6:.2f}MB -> {out_path}")