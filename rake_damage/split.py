import os, shutil, glob

base = "/home/claude/ds"
imgs = sorted(glob.glob(f"{base}/train/images/*"))
print(f"총 이미지: {len(imgs)}")

# 시간순 정렬되어 있으므로 뒤쪽 20%를 valid로 (인접 프레임 누수 최소화)
n_val = max(1, int(len(imgs)*0.2))
val_imgs = imgs[-n_val:]
train_imgs = imgs[:-n_val]

os.makedirs(f"{base}/valid/images", exist_ok=True)
os.makedirs(f"{base}/valid/labels", exist_ok=True)

def lbl_path(img_path, split):
    name = os.path.splitext(os.path.basename(img_path))[0]
    return f"{base}/{split}/labels/{name}.txt"

for img in val_imgs:
    name = os.path.basename(img)
    stem = os.path.splitext(name)[0]
    # 이미지 이동
    shutil.move(img, f"{base}/valid/images/{name}")
    # 라벨 이동
    src_lbl = f"{base}/train/labels/{stem}.txt"
    if os.path.exists(src_lbl):
        shutil.move(src_lbl, f"{base}/valid/labels/{stem}.txt")

print(f"train: {len(train_imgs)}장, valid: {len(val_imgs)}장")
