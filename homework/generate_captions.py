from pathlib import Path

import fire
import json
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    if not karts:
        return ["There are 0 karts in the scenario.", f"The track is {track_name}."]
    ego = [k for k in karts if k.get("is_center_kart")]
    ego = ego[0] if ego else karts[0]
    captions = []
    captions.append(f"{ego['kart_name']} is the ego car.")
    captions.append(f"There are {len(karts)} karts in the scenario.")
    captions.append(f"The track is {track_name}.")

    ego_center_x, ego_center_y = ego["center"]

    for kart in karts:
        if kart.get("is_center_kart"):
            continue
        kart_center_x, kart_center_y = kart["center"]
        kart_name = kart["kart_name"]

        if kart_center_x < ego_center_x:
            lr = "left"
            lr_alt = "to the left of"
        else:
            lr = "right"
            lr_alt = "to the right of"
        if kart_center_y < ego_center_y:
            fb = "ahead"
            fb_alt = "in front of"
        else:
            fb = "behind"
            fb_alt = "behind"

        captions.append(f"{kart_name} is {lr} and {fb} the ego car.")
        captions.append(f"{kart_name} is {lr} and {fb}")
        captions.append(f"{kart_name} {lr} {fb}")

    return captions
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.


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

def generate_all_captions(
    data_dir: str = "data/train",
    output_file: str = "data/train/generated_captions.json"
):
    data_dir = Path(data_dir)
    output_file = Path(output_file)

    info_files = sorted(data_dir.glob("*_info.json"))

    all_pairs = []

    print(f"Found {len(info_files)} info files in {data_dir}")

    for info_path in info_files:
        with open(info_path) as f:
            info = json.load(f)

        num_views = len(info["detections"])
        base_name = info_path.stem.replace("_info", "")

        for view_idx in range(num_views):

            # The corresponding image file
            image_file = f"{base_name}_{view_idx:02d}_im.jpg"
            image_path = data_dir / image_file

            if not image_path.exists():
                # Some datasets use PNG instead
                png_path = data_dir / f"{base_name}_{view_idx:02d}_im.png"
                if png_path.exists():
                    image_path = png_path
                else:
                    print(f"Warning: could not find image {image_file} for {info_path.name}")
                    continue

            captions = generate_caption(str(info_path), view_idx)

            # Flatten QA pairs into the required format
            for caption in captions:
                all_pairs.append(caption)
                all_pairs.append({
                    "caption": caption,
                    "image_file": "train/" + str(image_path.name)
                })

    # Save output
    with open(output_file, "w") as f:
        json.dump(all_pairs, f, indent=2)

    print(f"Saved {len(all_pairs)} QA pairs to {output_file}")

"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
        fire.Fire({
        "check": check_caption,
        "generate_all": generate_all_captions,
    })


if __name__ == "__main__":
    main()
