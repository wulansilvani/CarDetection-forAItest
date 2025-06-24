import os
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont

# === Konfigurasi Path ===
model_path = r"C:\Users\wulan\CarDetection\dataset_google_icrawler\resnet50_carclassifier_best.pth"
crop_dir = r"C:\Users\wulan\CarDetection\crops1"
output_img_dir = r"C:\Users\wulan\CarDetection\classified_crops"
unknown_dir = r"C:\Users\wulan\CarDetection\classified_crops_unknown"
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(unknown_dir, exist_ok=True)

# === Daftar kelas (8 kelas utama + Unknown secara manual)
class_names = [
    "MPV", "SUV", "Sedan", "Hatchback",
    "PickUp", "Minibus", "Truck", "Motorcycle", "Unknown"
]
num_model_classes = 9  # output dari model = 9

# === Load Model ResNet50 ===
model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, num_model_classes)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# === Transformasi input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Inference semua gambar
for filename in os.listdir(crop_dir):
    if not filename.lower().endswith(('.jpg', '.png')):
        continue

    img_path = os.path.join(crop_dir, filename)
    img = Image.open(img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        conf, pred = torch.max(probs, 0)

        # Aturan: jika confidence < 0.3 → Unknown
        if conf.item() < 0.3:
            predicted_class = "Unknown"
        else:
            predicted_class = class_names[pred.item()]
        
        label = f"{predicted_class} ({conf.item()*100:.1f}%)"

    # Tambahkan label ke gambar
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10), label, fill="yellow", font=font)

    # Simpan ke folder sesuai hasil
    if predicted_class == "Unknown":
        save_path = os.path.join(unknown_dir, filename)
    else:
        save_path = os.path.join(output_img_dir, filename)

    img.save(save_path)

print("✅ Semua gambar diklasifikasikan dan disimpan ke folder masing-masing.")
