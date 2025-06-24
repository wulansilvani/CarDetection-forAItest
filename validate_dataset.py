import os

image_dir = r"C:\Users\wulan\CarDetection\dataset\images"
label_dir = r"C:\Users\wulan\CarDetection\dataset\labels"

for subset in ['train', 'val', 'test']:
    img_path = os.path.join(image_dir, subset)
    lbl_path = os.path.join(label_dir, subset)

    img_files = {os.path.splitext(f)[0] for f in os.listdir(img_path) if f.endswith('.jpg')}
    lbl_files = {os.path.splitext(f)[0] for f in os.listdir(lbl_path) if f.endswith('.txt')}

    missing_txt = img_files - lbl_files
    missing_jpg = lbl_files - img_files

    print(f"\n=== {subset.upper()} ===")
    if missing_txt:
        print(f"Missing labels (.txt): {len(missing_txt)} files")
    if missing_jpg:
        print(f"Missing images (.jpg): {len(missing_jpg)} files")

    # Cek isi file .txt
    invalid_labels = 0
    for name in lbl_files:
        txt_path = os.path.join(lbl_path, f"{name}.txt")
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        invalid_labels += 1
                        break
                    # Cek semua angka (class_id boleh integer, lainnya float)
                    try:
                        int(parts[0])
                        [float(x) for x in parts[1:]]
                    except ValueError:
                        invalid_labels += 1
                        break
        except FileNotFoundError:
            print(f"File not found: {txt_path}")
            invalid_labels += 1

    print(f"Invalid label format: {invalid_labels} file(s)")
