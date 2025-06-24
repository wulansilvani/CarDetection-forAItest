import os
import json

# Path ke folder label JSON BDD100K
label_base_dir = r"C:\Users\wulan\CarDetection\bdd100k_labels\100k"
output_base_dir = r"C:\Users\wulan\CarDetection\yolo_labels"
image_width = 1280
image_height = 720


target_category = "car"
class_id = 0  # class id untuk YOLO, misalnya 'car' = 0

def convert_box(x1, y1, x2, y2, img_w, img_h):
    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return [x_center, y_center, width, height]

def process_json_folder(subset):
    label_folder = os.path.join(label_base_dir, subset)
    output_folder = os.path.join(output_base_dir, subset)
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(label_folder):
        if not file.endswith(".json"):
            continue
        file_path = os.path.join(label_folder, file)
        with open(file_path, "r") as f:
            data = json.load(f)

        frames = data.get("frames", [])
        if not frames:
            continue

        frame = frames[0]  # ambil 1 frame (karena hanya 1 gambar per file)
        objects = frame.get("objects", [])
        yolo_lines = []
        for obj in objects:
            if obj.get("category") != target_category:
                continue
            box2d = obj.get("box2d")
            if not box2d:
                continue
            x1, y1 = box2d["x1"], box2d["y1"]
            x2, y2 = box2d["x2"], box2d["y2"]
            bbox = convert_box(x1, y1, x2, y2, image_width, image_height)
            yolo_line = f"{class_id} {' '.join([f'{x:.6f}' for x in bbox])}"
            yolo_lines.append(yolo_line)

        # Simpan file .txt dengan nama yang sama
        output_txt = os.path.join(output_folder, f"{data['name']}.txt")
        with open(output_txt, "w") as out_f:
            out_f.write("\n".join(yolo_lines))

        print(f"Converted: {file} → {output_txt}")


for subset in ["train", "val", "test"]:
    process_json_folder(subset)

print("✅ All labels converted to YOLO format.")
