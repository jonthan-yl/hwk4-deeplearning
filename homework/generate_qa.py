import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).
    """
    with open(info_path) as f:
        info = json.load(f)

    detections_per_view = info["detections"]
    if view_index >= len(detections_per_view):
        return []

    detections = detections_per_view[view_index]

    # Compute scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    karts = []

    for det in detections:
        class_id, instance_id, x1, y1, x2, y2 = det

        if int(class_id) != 1:
            continue  # only karts

        # Scale box
        x1s = x1 * scale_x
        y1s = y1 * scale_y
        x2s = x2 * scale_x
        y2s = y2 * scale_y

        # Skip small / invalid boxes
        if (x2s - x1s) < min_box_size or (y2s - y1s) < min_box_size:
            continue
        if x2s < 0 or x1s > img_width or y2s < 0 or y1s > img_height:
            continue

        cx = (x1s + x2s) / 2
        cy = (y1s + y2s) / 2

        kart_name = info["instances"][str(instance_id)]["name"] if "instances" in info else f"kart_{instance_id}"

        karts.append({
            "instance_id": instance_id,
            "kart_name": kart_name,
            "center": (cx, cy),
        })

    # Identify ego car = kart closest to image center
    img_center = (img_width / 2, img_height / 2)

    def dist_to_center(k):
        cx, cy = k["center"]
        return (cx - img_center[0])**2 + (cy - img_center[1])**2

    if karts:
        ego = min(karts, key=dist_to_center)
        for k in karts:
            k["is_center_kart"] = (k is ego)
    else:
        pass

    return karts



def extract_track_info(info_path: str) -> str:
    """
    Extract track name from info.json.
    """
    with open(info_path) as f:
        info = json.load(f)
    print(info)
    if "track" in info:
        return info["track"]

    return "Unknown Track"



def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.
    """
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    if not karts:
        return [{"question": "How many karts are visible?", "answer": "0"}]

    # Identify ego kart
    ego = [k for k in karts if k.get("is_center_kart")]
    ego = ego[0] if ego else karts[0]

    ego_x, ego_y = ego["center"]

    qa = []

    # 1. Ego car question
    qa.append({
        "question": "What kart is the ego car?",
        "answer": ego["kart_name"]
    })

    # 2. Total karts
    qa.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(len(karts))
    })

    # 3. Track info
    qa.append({
        "question": "What track is this?",
        "answer": track_name
    })

    # 4. Relative position questions
    for k in karts:
        if k["instance_id"] == ego["instance_id"]:
            continue

        name = k["kart_name"]
        x, y = k["center"]

        horizontal = "left" if x < ego_x else "right"
        vertical = "in front" if y < ego_y else "behind"

        qa.append({
            "question": f"Is {name} to the left or right of the ego car?",
            "answer": horizontal
        })
        qa.append({
            "question": f"Is {name} in front of or behind the ego car?",
            "answer": "in front" if y < ego_y else "behind"
        })
        qa.append({
            "question": f"Where is {name} relative to the ego car?",
            "answer": f"{vertical} and to the {horizontal}"
        })

    # 5. Counting questions
    left_count = sum(1 for k in karts if k["center"][0] < ego_x and k != ego)
    right_count = sum(1 for k in karts if k["center"][0] > ego_x and k != ego)
    front_count = sum(1 for k in karts if k["center"][1] < ego_y and k != ego)
    behind_count = sum(1 for k in karts if k["center"][1] > ego_y and k != ego)

    qa.append({"question": "How many karts are to the left of the ego car?", "answer": str(left_count)})
    qa.append({"question": "How many karts are to the right of the ego car?", "answer": str(right_count)})
    qa.append({"question": "How many karts are in front of the ego car?", "answer": str(front_count)})
    qa.append({"question": "How many karts are behind the ego car?", "answer": str(behind_count)})

    return qa


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)

    def generate_all_qa_pairs(
        data_dir: str = "data/train",
        output_file: str = "data/train/generated_qa_pairs.json"
    ):
        """
        Iterate through all *_info.json files in data/train/ and generate QA pairs
        for every view inside each file. Save all results into one JSON file.

        Args:
            data_dir: Directory containing *_info.json files.
            output_file: Where to save the final JSON list.
        """

        data_dir = Path(data_dir)
        output_file = Path(output_file)

        info_files = sorted(data_dir.glob("*_info.json"))

        all_pairs = []

        print(f"Found {len(info_files)} info files in {data_dir}")
        counter = 0
        for info_path in info_files:
            counter += 1
            if counter % 100 == 0:
                counter = 0
                print(f"Processing file {counter}/{len(info_files)}: {info_path.name}")
            with open(info_path) as f:
                info = json.load(f)

            num_views = len(info["detections"])
            base_name = info_path.stem.replace("_info", "")

            print(f"Processing {info_path.name} with {num_views} views")

            for view_idx in range(num_views):
                qa_pairs = generate_qa_pairs(str(info_path), view_idx)

                all_pairs.append({
                    "file": info_path.name,
                    "base_image_id": base_name,
                    "view_index": view_idx,
                    "qa_pairs": qa_pairs
                })

        # Save final result
        with open(output_file, "w") as f:
            json.dump(all_pairs, f, indent=2)

        print(f"Saved {len(all_pairs)} QA entry sets to {output_file}")



"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_qa_pairs})


if __name__ == "__main__":
    main()
