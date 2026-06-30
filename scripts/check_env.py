import torch

print("=== CUDA 환경 확인 ===")
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU 장치 수: {torch.cuda.device_count()}")
    print(f"현재 GPU (device=0): {torch.cuda.get_device_name(0)}")
    print(f"CUDA 버전: {torch.version.cuda}")
else:
    print("[경고] CUDA를 사용할 수 없습니다. CPU로 폴백됩니다.")
