import os

def count_images(folder_path):
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                count += 1
    return count

base_path = "P2datasetFull"

folders = {
    "Train Adult": f"{base_path}/train/adult",
    "Train Non-Adult": f"{base_path}/train/non_adult",
    "Val Adult": f"{base_path}/val/adult",
    "Val Non-Adult": f"{base_path}/val/non_adult",
}

print("📊 DATASET IMAGE COUNTS\n")

for name, path in folders.items():
    if os.path.exists(path):
        print(f"{name}: {count_images(path)} images")
    else:
        print(f"{name}: ❌ Folder not found -> {path}")
