# 제진기 레이크 파손 검출 (YOLOv8-seg)

## 프로젝트 목표
제진기(screen rake) 영상에서 레이크 파손을 **인스턴스 세그멘테이션**으로 검출하고,
파손 영역을 픽셀 면적으로 정량화한다.

- 모델: Ultralytics YOLOv8-seg (`yolov8s-seg` 기준, 정밀도 필요시 `m-seg`)
- 학습 환경: RTX 5070 Ti (CUDA, device=0)
- 클래스: `0: bent(휘어짐)`, `1: broken(부러짐)`, `2: missing(결손)`
- 데이터: 합성 이미지(+합성 마스크) 위주, 실데이터 일부를 val에 혼합

## 핵심 원칙
1. **합성 마스크를 자동으로 YOLO-seg 라벨로 변환하는 것이 1순위.** 수작업 라벨링 최소화.
2. 합성에서만 잘 되고 실데이터에서 무너지는 도메인 갭을 항상 모니터링. 실데이터 프레임을 반드시 val에 포함.
3. 파손 형태별로 클래스를 나눠 검출 후 활용도(심각도 판단)를 높인다.

---

## 디렉터리 구조 (목표)
```
rake_seg/
├── synth/                 # 합성 단계 산출물
│   ├── images/            # 합성 이미지 (png/jpg)
│   └── masks/             # 클래스별 이진 마스크 (파일명 규칙: <name>_<classid>.png)
├── dataset/
│   ├── images/{train,val}/
│   ├── labels/{train,val}/   # YOLO-seg 폴리곤 txt
│   └── data.yaml
├── scripts/
│   ├── mask_to_yoloseg.py    # 합성 마스크 → YOLO-seg 라벨 변환
│   ├── split_dataset.py      # train/val 분할 (+ 실데이터 val 혼합)
│   ├── train.py
│   └── infer_video.py        # 영상 추론 + 파손 면적 정량화
├── runs/                  # 학습 결과 (자동 생성)
└── requirements.txt
```

---

## TASK 1 — 환경 셋업
- `requirements.txt` 작성: `ultralytics`, `opencv-python`, `numpy`, `pyyaml`
- CUDA 사용 가능 여부 확인 코드 포함 (`torch.cuda.is_available()`, device 이름 출력)
- 설치 후 `yolo checks` 실행으로 GPU 인식 검증

## TASK 2 — 합성 마스크 → YOLO-seg 변환 (`scripts/mask_to_yoloseg.py`)
요구사항:
- 입력: `synth/images/`, `synth/masks/` (마스크는 이진, 파일명에 class_id 포함)
- 마스크에서 `cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`로 폴리곤 추출
- 면적 임계값(`min_area`, 기본 50px)으로 노이즈 제거
- 폴리곤이 너무 촘촘하면 `cv2.approxPolyDP`로 점 개수 축소 (epsilon 인자화)
- 좌표를 이미지 크기로 0~1 정규화
- 출력 포맷: `class_id x1 y1 x2 y2 ... xn yn` (한 객체당 한 줄)
- 한 이미지에 여러 객체/여러 클래스 → 여러 줄로 누적
- 폴리곤 점이 6개 미만(=3쌍 미만)이면 스킵 (유효 폴리곤 아님)
- CLI 인자: 입력/출력 경로, min_area, epsilon
- 변환 후 라벨 1~2개를 원본 위에 그려서 `_debug/` 에 시각화 저장 (검증용)

## TASK 3 — 데이터 분할 (`scripts/split_dataset.py`)
- 합성 데이터를 train/val로 분할 (기본 9:1)
- **실데이터 프레임이 있으면 전량 val로** 배치 (도메인 갭 측정용), 옵션으로 비율 조정
- 이미지/라벨 쌍 무결성 검사 (라벨 없는 이미지, 이미지 없는 라벨 경고)
- `data.yaml` 자동 생성 (path/train/val/names)

## TASK 4 — 학습 (`scripts/train.py`)
- `YOLO("yolov8s-seg.pt")` 로 시작, 모델 크기 CLI 인자화
- 파라미터: `epochs=100, imgsz=640, batch=16, device=0`
- 데이터 증강은 기본값 유지하되, 합성↔실제 갭 완화용으로 `hsv_*`, `degrees`, `scale` 가볍게 조정 가능하게 인자화
- 학습 후 best.pt 경로 출력, val 지표(mAP50-95(M)) 콘솔 요약

## TASK 5 — 영상 추론 + 정량화 (`scripts/infer_video.py`)
- 입력 영상 프레임 단위 추론 (`conf` 인자화, 기본 0.4)
- 각 인스턴스: 클래스, confidence, **마스크 픽셀 면적**(`mask.sum()`) 계산
- 프레임별 파손 면적 합계를 CSV로 로깅 (frame_idx, class, area_px, conf)
- 결과 영상(`results.mp4`) 저장: 마스크 오버레이 + 클래스/면적 텍스트
- 면적 임계 초과 시 해당 프레임을 `alerts/` 에 따로 저장 (심각 파손 후보)

---

## 검증 체크리스트
- [ ] TASK 2 변환 후 `_debug/` 시각화에서 폴리곤이 파손부에 정확히 맞는가
- [ ] data.yaml의 names 순서와 마스크 파일명 class_id가 일치하는가
- [ ] 학습 시 GPU(device=0)가 실제로 사용되는가 (CPU 폴백 아님)
- [ ] val에 실데이터가 포함되어 합성↔실제 갭이 지표로 보이는가
- [ ] 추론 CSV의 area_px가 파손 심각도와 단조 증가 관계인가

## 주의
- 마스크 파일명-class_id 매핑 규칙을 코드 상단 상수로 명시하고 바꾸기 쉽게.
- 좌표 정규화에서 이미지 w/h를 마스크가 아닌 **실제 이미지 크기**로 사용.
- `cv2.destroyAllWindows()` 등 GUI 호출은 헤드리스 환경 고려해 옵션 처리.
