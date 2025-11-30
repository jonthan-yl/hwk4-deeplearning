from pathlib import Path
import json

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    # find ego
    ego = None
    for k in karts:
        if k.get("is_center_kart"):
            ego = k
            break

    captions = []
    if ego is not None:
        captions.append(f"{ego['kart_name']} is the ego car.")

    captions.append(f"There are {len(karts)} karts in the scene.")
    if track_name is not None:
        captions.append(f"The track is {track_name}.")

    # Add at most 2 relative position captions for variety
    count = 0
    for k in karts:
        if ego is None or k["instance_id"] == ego["instance_id"]:
            continue
        kx, ky = k["center"]
        ego_x, ego_y = (ego["center"] if ego is not None else (img_width / 2.0, img_height / 2.0))

        lr = "right" if kx > ego_x else "left"
        fb = "front" if ky < ego_y else "back"
        captions.append(f"{k['kart_name']} is {fb} and {lr} of the ego car.")
        count += 1
        if count >= 2:
            break

    return captions


def generate_dataset(split: str = "train", output_file: str | None = None, data_dir: str | None = None, max_files: int | None = None, max_views_per_file: int | None = None) -> str:
    root_data_dir = Path(data_dir) if data_dir is not None else (Path(__file__).parent.parent / "data")
    split_dir = root_data_dir / split
    if output_file is None:
        output_file = split_dir / "balanced_captions.json"
    else:
        output_file = Path(output_file)

    info_files = sorted(split_dir.glob("*_info.json"))
    if max_files is not None:
        info_files = info_files[:max_files]

    all_captions = []
    for info_path in info_files:
        with open(info_path) as f:
            info = json.load(f)
        num_views = len(info.get("detections", []))
        views_to_process = range(num_views if max_views_per_file is None else min(num_views, max_views_per_file))
        for v in views_to_process:
            captions = generate_caption(str(info_path), v)
            base_name = Path(info_path).stem.replace("_info", "")
            image_file = list(Path(info_path).parent.glob(f"{base_name}_{v:02d}_im.jpg"))
            image_file_str = f"{split}/{image_file[0].name}" if image_file else f"{split}/{base_name}_{v:02d}_im.jpg"
            for cap in captions:
                all_captions.append({"image_file": image_file_str, "caption": cap})

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as out_f:
        json.dump(all_captions, out_f, indent=2)
    print(f"Wrote {len(all_captions)} captions to {output_file}")
    return str(output_file)


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "generate": generate_dataset})


if __name__ == "__main__":
    main()
