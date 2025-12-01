import json
from pathlib import Path
import fire
from matplotlib import pyplot as plt
from generate_qa import draw_detections, extract_kart_objects, extract_track_info, extract_frame_info

def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate captions for a specific view.
    """
    captions = []

    visible_karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    if not visible_karts:
        return captions

    # Ego kart
    ego_kart = next((k for k in visible_karts if k.get("is_center_kart")), None)
    num_karts = len(visible_karts)
    track_name = extract_track_info(info_path)

    # 1. Ego car
    if ego_kart:
        captions.append(f"{ego_kart['name']} is the ego car.")

    # 2. Counting
    captions.append(f"There are {num_karts} karts in the scenario.")

    # 3. Track name
    captions.append(f"The track is {track_name}.")

    # 4. Relative positions
    if ego_kart:
        ego_x, ego_y = ego_kart["center"]
        for kart in visible_karts:
            if kart["track_id"] == ego_kart["track_id"]:
                continue

            dx = kart["center"][0] - ego_x
            dy = kart["center"][1] - ego_y

            lr = "left" if dx < 0 else "right"
            fb = "front" if dy < 0 else "behind"

            captions.append(f"{kart['name']} is to the {lr} of the ego car.")
            captions.append(f"{kart['name']} is {fb} the ego car.")
            captions.append(f"{kart['name']} is {fb} {lr} of the ego car.")

    return captions

def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)
    print("\nCaptions:")
    print("-" * 50)
    for i, c in enumerate(captions):
        print(f"{i+1}. {c}")
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

def generate_dataset(data_dir: str = "data/train", output_file: str = "train_captions.json"):
    """
    Generate massive CLIP training dataset (image_file, caption)
    """
    data_path = Path(data_dir)
    all_data = []

    info_files = sorted(list(data_path.glob("*_info.json")))
    print(f"Found {len(info_files)} info files.")

    for info_file in info_files:
        with open(info_file, 'r') as f:
            data = json.load(f)
            num_views = len(data.get("detections", []))

        frame_id_str = info_file.stem.split("_")[0]

        for view_idx in range(num_views):
            image_name = f"{frame_id_str}_{view_idx:02d}_im.jpg"
            image_path = info_file.parent / image_name
            if not image_path.exists():
                continue

            captions = generate_caption(str(info_file), view_idx)
            rel_image_path = f"{data_path.name}/{image_name}"

            for c in captions:
                all_data.append({"image_file": rel_image_path, "caption": c})

    print(f"Total captions generated: {len(all_data)}")
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved dataset to {output_file}")

def main():
    fire.Fire({
        "check": check_caption,
        "generate": generate_dataset
    })

if __name__ == "__main__":
    main()
