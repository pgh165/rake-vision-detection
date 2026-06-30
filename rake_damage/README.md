# 제진기 레이크 파손 검출 (YOLOv8n)

제진기(screen rake) 영상에서 **레이크 파손(damage)** 을 객체 검출(bounding box)로 찾아낸다.

## 개요
- **태스크**: 객체 검출 (Object Detection, bbox)
- **모델**: Ultralytics `yolov8n` 전이학습
- **클래스**: `0: damage` (단일 클래스)
- **입력 크기**: `imgsz=640`
- **학습 환경**: RTX 5070 Ti (CUDA, `device=0`)

## 데이터셋
- 출처: Roboflow Universe (`my-first-project-iz8yo`, CC BY 4.0) — [README.dataset.txt](README.dataset.txt), [README.roboflow.txt](README.roboflow.txt)
- 구성: `train/` 38장, `valid/` 8장
- 설정: [data.yaml](data.yaml)

```
rake_damage/
├── data.yaml        # 데이터셋 경로 / 클래스 정의
├── train/images,labels  # 학습 세트 (38)
├── valid/images,labels  # 검증 세트 (8)
├── split.py         # 시간순 뒤 20%를 valid로 분리 (프레임 누수 방지)
├── train.py         # 학습 스크립트
└── make.py          # 영상 추론 + 검출 박스 오버레이 영상 생성
```

## 데이터 분할 — `split.py`
영상에서 추출한 프레임은 인접 프레임끼리 거의 동일하다. 무작위 분할 시 **train/valid 누수**가
생기므로, **시간순 정렬 후 뒤쪽 20% 를 valid** 로 떼어내 누수를 최소화한다.

## 학습 — `train.py`
`yolov8n.pt` 사전학습 가중치에서 시작하여 전이학습한다.

```bash
python train.py
```

주요 하이퍼파라미터
| 항목 | 값 | 비고 |
|------|-----|------|
| epochs | 50 | val 곡선상 10~15 정점 → 50이면 충분, `patience=10` |
| imgsz | 640 | |
| batch | 16 | |
| 증강 | hsv/degrees/translate/scale/fliplr/mosaic 약하게 | 데이터 적음(38장) → 과적합 억제 |

학습 후 `mAP50`, `mAP50-95` 를 출력한다. 결과 가중치: `rake_damage/exp_clean/weights/best.pt`

## 추론 영상 생성 — `make.py`
학습된 `best.pt` 로 영상에 파손 박스를 오버레이해 결과 영상을 만든다.
- `conf=0.15`
- 프레임마다 검출 개수(`damage: N`)를 좌상단에 표기
- **코덱 폴백**(`mp4v → avc1 → XVID/avi`) + VideoWriter 열림 검증으로 0바이트 출력 방지

```bash
python make.py   # 상단 MODEL_PATH / VIDEO_IN 경로 수정 후 실행
```

## 결과물
- `result_detection.mp4` — 파손 검출 박스 오버레이 영상 (용량상 제외)
- `2001189 박지호 레이크 파손.mp4` — 시연 영상 (제외)

> 영상·모델 가중치(`*.mp4`, `*.pt`)는 용량 때문에 `.gitignore` 처리되어 있다.
> 학습을 직접 돌리면 `best.pt` 가 재생성된다.
