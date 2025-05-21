import os
import json
from pathlib import Path

def generate_meta(dataset_root, image_exts=('.png', '.jpg', '.jpeg'), mask_exts=('.png', '.jpg', '.jpeg')):
    import os
    import json
    from pathlib import Path

    meta = {"train": {}, "test": {}}
    dataset_root = Path(dataset_root)

    def is_valid(file, exts):
        return file.suffix.lower() in exts

    # Train section
    train_dir = dataset_root / "train"
    if train_dir.exists():
        for cls_dir in train_dir.iterdir():
            cls_name = cls_dir.name
            samples = []
            good_dir = cls_dir / "good"
            if not good_dir.exists():
                continue
            for img_path in sorted(good_dir.glob("*")):
                if not is_valid(img_path, image_exts): continue
                samples.append({
                    "img_path": str(img_path.relative_to(dataset_root)),
                    "mask_path": "",
                    "cls_name": cls_name,
                    "specie_name": "good",
                    "anomaly": 0
                })
            meta["train"][cls_name] = samples

    # Test section
    test_dir = dataset_root / "test"
    if test_dir.exists():
        for cls_dir in test_dir.iterdir():
            cls_name = cls_dir.name
            samples = []
            for specie_dir in cls_dir.iterdir():
                if specie_dir.name == "masks" or not specie_dir.is_dir():
                    continue
                for img_path in sorted(specie_dir.glob("*")):
                    if not is_valid(img_path, image_exts): continue
                    anomaly = 0 if specie_dir.name.lower() == "good" else 1
                    mask_path = ""
                    if anomaly == 1:
                        mask_dir = cls_dir / "masks"
                        for ext in mask_exts:
                            candidate = mask_dir / f"{img_path.stem}_mask{ext}"
                            if candidate.exists():
                                mask_path = str(candidate.relative_to(dataset_root))
                                break
                    samples.append({
                        "img_path": str(img_path.relative_to(dataset_root)),
                        "mask_path": mask_path,
                        "cls_name": cls_name,
                        "specie_name": specie_dir.name,
                        "anomaly": anomaly
                    })
            meta["test"][cls_name] = samples

    # Save
    out_path = dataset_root / "meta.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"✅ meta.json generated at {out_path}")
