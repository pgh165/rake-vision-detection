# 제진기 비전 검출 (Screen Rake Vision)

제진기(스크린 레이크) 설비를 대상으로 한 두 가지 YOLO 비전 검출 프로젝트.
RTX 5070 Ti 에서 학습하며, 입력 크기는 `imgsz=640` 으로 통일했다.

## 구성

| 폴더 | 프로젝트 | 태스크 | 모델 | 클래스 | 데이터(train/valid) |
|------|----------|--------|------|--------|---------------------|
| [`debris_detection/`](debris_detection/README.md) | **부유물 검출** | 인스턴스 세그멘테이션 | `yolov8n-seg` | `debris` | 80 / 16 |
| [`rake_damage/`](rake_damage/README.md) | **레이크 파손 검출** | 객체 검출 (bbox) | `yolov8n` | `damage` | 383 / 66 |

각 프로젝트의 상세 학습/추론 절차는 폴더 안의 `README.md` 참고.

### 1. 부유물 검출 (`debris_detection`)
수면 위 부유물을 세그멘테이션으로 검출하고, 마스크들을 하나의 덩어리로 통합한 뒤
**화면 면적 비율 + 시간 평활화**로 "적음 / 보통 / 많음(청소 필요)" 단계를 판정한다.
→ [상세 문서](debris_detection/README.md)

### 2. 레이크 파손 검출 (`rake_damage`)
레이크 파손부를 bbox 로 검출한다. 영상 프레임 누수를 막는 **시간순 분할**과
결과 영상 생성까지 포함한다.
→ [상세 문서](rake_damage/README.md)

## 공통 환경
- GPU: RTX 5070 Ti (CUDA, `device=0`)
- 프레임워크: Ultralytics YOLOv8
- 의존성: [requirements.txt](requirements.txt) (`ultralytics`, `opencv-python`, `numpy`, `pyyaml`)
- 데이터셋 출처: Roboflow Universe (CC BY 4.0)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## 저장소에서 제외된 항목
용량 문제로 다음은 `.gitignore` 처리되어 있다 (학습/추론을 직접 실행하면 재생성됨).
- 영상 파일 `*.mp4`, `*.avi` (시연·결과 영상)
- 모델 가중치 `*.pt`, `*.onnx`
- 학습 산출물 `runs/`, 가상환경 `venv/`, 라벨 캐시 `*.cache`
