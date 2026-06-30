# 제진기 부유물 검출 (YOLOv8n-seg)

제진기(screen) 영상에서 수면 위 **부유물(debris)** 을 인스턴스 세그멘테이션으로 검출하고,
화면 내 부유물 면적 비율을 정량화하여 "적음 / 보통 / 많음(청소 필요)" 단계를 판정한다.

## 개요
- **태스크**: 인스턴스 세그멘테이션 (Instance Segmentation)
- **모델**: Ultralytics `yolov8n-seg` 전이학습
- **클래스**: `0: debris` (단일 클래스)
- **입력 크기**: `imgsz=640` (Hailo-8L 엣지 추론 호환)
- **학습 환경**: RTX 5070 Ti (CUDA, `device=0`)

## 데이터셋
- 출처: Roboflow Universe (`debris-o63iw`, CC BY 4.0) — 자세한 내용은 [README.dataset.txt](README.dataset.txt), [README.roboflow.txt](README.roboflow.txt)
- 구성: `train/` 80장, `valid/` 16장
- 설정: [data.yaml](data.yaml)

```
debris_detection/
├── data.yaml            # 데이터셋 경로 / 클래스 정의
├── train/images,labels  # 학습 세트 (80)
├── valid/images,labels  # 검증 세트 (16)
├── train_seg.py         # 학습 스크립트
└── judge_debris.py      # 영상 추론 + 부유물 양 판정
```

## 학습 — `train_seg.py`
`yolov8n-seg.pt` 사전학습 가중치에서 시작하여 100 epoch 전이학습한다.

```bash
python train_seg.py
```

주요 하이퍼파라미터
| 항목 | 값 | 비고 |
|------|-----|------|
| epochs | 100 | `patience=20` 조기종료 |
| imgsz | 640 | Hailo-8L 호환 |
| batch | 16 | |
| 증강 | hsv/degrees/translate/scale/fliplr/mosaic | 데이터 다양성 양호(80장) → 표준 수준 |

학습 후 Box / Mask 각각의 `mAP50`, `mAP50-95` 를 콘솔에 출력한다.
결과 가중치: `debris_seg/exp1/weights/best.pt`

## 추론 + 정량화 — `judge_debris.py`
개별 마스크를 그대로 쓰지 않고 **하나의 부유물 덩어리로 통합**하여 면적 비율로 판정한다.

처리 파이프라인
1. 프레임별 세그멘테이션 추론 (`conf=0.15`, `retina_masks=True`)
2. 모든 인스턴스 마스크를 원본 해상도로 OR 결합
3. 영상 해상도로 **전체 픽셀 수** `TOTAL_PX = W*H` 계산 (비율 분모)
4. **인식된 세그멘테이션 픽셀 수** `seg_px = union.sum()` 를 전체 픽셀 대비 비율(%)로 환산
   → `pct = seg_px / TOTAL_PX * 100`
5. 최근 `SMOOTH_N=15` 프레임 **시간 평활화** 로 수치 안정화
6. 면적 비율(%)로 단계 판정
   - `< 10%` → **CLEAR**
   - `10~30%` → **LOW**
   - `30~60%` → **MEDIUM**
   - `60~100%` → **HIGH** (청소 필요)
7. 통합 덩어리(`dilate`+`CLOSE`, `DILATE=35`)를 반투명 오버레이 + 외곽선으로 표시,
   면적%·단계·덩어리 수를 패널에 출력 → 결과 영상 저장

```bash
python judge_debris.py   # 상단 MODEL_PATH / VIDEO_IN 경로 수정 후 실행
```

설정 상수 (`judge_debris.py` 상단)
| 상수 | 기본값 | 의미 |
|------|--------|------|
| `CONF` | 0.15 | 마스크 감지 임계값 |
| `DILATE` | 35 | 덩어리 통합 강도 (20~50) |
| `SMOOTH_N` | 15 | 시간 평활화 프레임 수 |
| `TH_LOW` / `TH_MEDIUM` / `TH_HIGH` | 10 / 30 / 60 | LOW/MEDIUM/HIGH 면적 비율(%) 임계 |

## 결과물
- `result_debris.mp4` — 부유물 영역 오버레이 + 단계 판정 영상 (용량상 저장소에서 제외)
- `2001189 박지호 부유물 검출.mp4` — 시연 영상 (제외)

> **데이터셋**(`train/`, `valid/`)과 영상·모델 가중치(`*.mp4`, `*.pt`)는 `.gitignore` 처리되어
> 저장소에 없다. 데이터셋은 위 Roboflow 링크에서 받고, 학습을 직접 돌리면 `best.pt` 가 재생성된다.
