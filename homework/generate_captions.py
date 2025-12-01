from pathlib import Path
import fire
from matplotlib import pyplot as plt

from generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info

def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate captions for a specific view.
    """
    captions = []

    # Extract karts and ego
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    if not karts:
        return captions

    ego_kart = next(k for k in karts if k.get("is_center_kart"))
    num_karts = len(karts)
    track_name = extract_track_info(info_path)

    # 2. Counting
    captions.append(f"There are {num_karts} karts in the scenario.")

    # 3. Track name
    captions.append(f"The track is {track_name}.")

    # 4. Relative position of other karts
    for kart in karts:
        if kart["is_center_kart"]:
            continue
        dx = kart["center"][0] - ego_kart["center"][0]
        dy = kart["center"][1] - ego_kart["center"][1]

        lr = "left" if dx < 0 else "right"
        fb = "front" if dy < 0 else "behind"

        captions.append(f"{kart['kart_name']} is to the {lr} of the ego car.")
        captions.append(f"{kart['kart_name']} is {fb} the ego car.")

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaptions:")
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


def main():
    fire.Fire({"check": check_caption})


if __name__ == "__main__":
    main()
