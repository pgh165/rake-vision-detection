# 제진기 레이크 파손 검출 (YOLOv8n)

제진기(screen rake) 영상에서 **레이크 파손(damage)** 을 객체 검출(bounding box)로 찾아낸다.

## 개요
- **태스크**: 객체 검출 (Object Detection, bbox)
- **모델**: Ultralytics `yolov8n` 전이학습
- **클래스**: `0: damage` (단일 클래스)
- **입력 크기**: `imgsz=640`
- **학습 환경**: RTX 5070 Ti (CUDA, `device=0`)

## 데이터셋 (v2, 2026-06-30 갱신)
- 출처: Roboflow Universe (`my-first-project-iz8yo`, CC BY 4.0) — [README.dataset.txt](README.dataset.txt), [README.roboflow.txt](README.roboflow.txt)
- 구성: `train/` **383장** (고유 원본 115개), `valid/` **66장** (고유 원본 22개) — 총 449장
- 영상에서 파손이 뚜렷한 프레임(`dmg_*`, `KakaoTalk_*`)을 추가 추출 → Roboflow에서 **증강**(원본 1장당 여러 장 생성)
- 분할: Roboflow에서 **그룹 단위 분할** — 같은 원본의 증강본이 train/valid 에 섞이지 않게 하여 **증강 누수 차단**
- 설정: [data.yaml](data.yaml)

```
rake_damage/
├── data.yaml        # 데이터셋 경로 / 클래스 정의
├── train/images,labels  # 학습 세트 (383, 증강 포함)
├── valid/images,labels  # 검증 세트 (66)
├── split.py         # (구 v1) 시간순 뒤 20%를 valid로 분리하던 스크립트
├── train.py         # 학습 스크립트
└── make.py          # 영상 추론 + 검출 박스 오버레이 영상 생성
```

## 데이터 분할
v2 데이터셋은 **Roboflow에서 그룹 단위로 분할**된 채 내려받는다. 같은 원본 프레임에서 파생된
증강본이 train/valid 양쪽에 흩어지면 검증 점수가 부풀려지는 **증강 누수**가 생기므로, 원본 그룹째로
한쪽에만 배치한다.
> `split.py` 는 v1에서 영상 프레임을 시간순 뒤 20%로 잘라 valid로 떼던 스크립트로, v2에서는 사용하지 않는다.

## 학습 — `train.py`
`yolov8n.pt` 사전학습 가중치에서 시작하여 전이학습한다.

```bash
python train.py
```

주요 하이퍼파라미터 (v2)
| 항목 | 값 | 비고 |
|------|-----|------|
| epochs | 100 | `patience=20` 조기종료 |
| imgsz | 640 | |
| batch | 16 | |
| 증강 | hsv/degrees/translate/scale/fliplr/mosaic 약하게 | Roboflow에서 이미 증강됨 → 학습측 증강은 약하게 |

학습 후 `mAP50`, `mAP50-95` 를 출력한다.
결과 가중치: `rake_damage/runs/detect/rake_damage_v2/exp1/weights/best.pt`

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

> **데이터셋**(`train/`, `valid/`)과 영상·모델 가중치(`*.mp4`, `*.pt`)는 `.gitignore` 처리되어
> 저장소에 없다. 데이터셋은 위 Roboflow 링크에서 받고, 학습을 직접 돌리면 `best.pt` 가 재생성된다.
