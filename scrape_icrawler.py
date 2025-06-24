from icrawler.builtin import GoogleImageCrawler
import os

categories = {
#    "MPV": ["Avanza", "Xenia", "Ertiga"],
#    "SUV": ["Rush", "Terios", "CR-V", "Fortuner"],
#    "Sedan": ["Vios", "Lancer", "Civic"],
#    "Hatchback": ["Yaris", "Jazz", "Brio"],
#    "PickUp": ["L300", "Grand Max Pick-up"],
#    "Truck": ["Colt Diesel"],
#    "Minibus": ["ELF", "HiAce"],

  "Motorcycle": [
    "Honda Beat",
    "Honda Vario",
    "Yamaha Mio",
    "Yamaha NMAX",
    "Honda Scoopy",
    "Yamaha Aerox",
    "Honda Supra X",
    "Honda Revo",
    "Yamaha Vega R",
    "Suzuki Satria FU"
]
}

base_dir = "dataset_google_icrawler"
max_images = 10

for cat, models in categories.items():
    for model in models:
        save_dir = os.path.join(base_dir, cat, model.replace(" ", "_"))
        os.makedirs(save_dir, exist_ok=True)

        google_crawler = GoogleImageCrawler(storage={'root_dir': save_dir})
        google_crawler.crawl(keyword=model + " car", max_num=max_images)
        print(f"[DONE] {model} → gambar disimpan di {save_dir}")
