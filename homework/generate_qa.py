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
        # Some datasets use hex frame id — keep previous behaviour
        try:
            frame_id = int(parts[0], 16)  # Convert hex to decimal
        except Exception:
            try:
                frame_id = int(parts[0])
            except Exception:
                frame_id = 0
        try:
            view_index = int(parts[1])
        except Exception:
            view_index = 0
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

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center (scaled to img_width/img_height)
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """
    with open(info_path) as f:
        info = json.load(f)

    # defensive checks
    if "detections" not in info:
        raise KeyError("info.json missing 'detections' key")

    if view_index >= len(info["detections"]):
        raise IndexError(f"view_index {view_index} out of range in info.json detections")

    detections = info["detections"][view_index]

    # Attempt to load names for instances if present
    instance_names = {}
    # known possible keys that might hold names
    for k in ("instances", "objects", "labels", "names"):
        if k in info and isinstance(info[k], dict):
            # assume mapping id->name or str keys
            for tid, val in info[k].items():
                try:
                    instance_names[int(tid)] = str(val)
                except Exception:
                    # if keys are not ints, try parsing
                    try:
                        instance_names[int(val.get("id"))] = val.get("name", f"kart_{val.get('id')}")
                    except Exception:
                        continue

    # fallback: sometimes the info contains a list of objects with properties
    if not instance_names and "objects" in info and isinstance(info["objects"], list):
        for obj in info["objects"]:
            try:
                tid = int(obj.get("track_id", obj.get("id", -1)))
                name = obj.get("name") or obj.get("label")
                if tid >= 0 and name:
                    instance_names[tid] = name
            except Exception:
                continue

    kart_list = []
    for detection in detections:
        # detection format: [class_id, track_id, x1, y1, x2, y2]
        try:
            class_id, track_id, x1, y1, x2, y2 = detection
            class_id = int(class_id)
            track_id = int(track_id)
        except Exception:
            # more defensive parsing
            if len(detection) >= 6:
                class_id = int(detection[0])
                track_id = int(detection[1])
                x1, y1, x2, y2 = map(float, detection[2:6])
            else:
                continue

        if class_id != 1:
            continue  # only karts

        # scale coordinates from ORIGINAL_WIDTH/HEIGHT to desired img_width/img_height
        scale_x = img_width / ORIGINAL_WIDTH
        scale_y = img_height / ORIGINAL_HEIGHT
        cx = (x1 + x2) / 2.0 * scale_x
        cy = (y1 + y2) / 2.0 * scale_y
        w = (x2 - x1) * scale_x
        h = (y2 - y1) * scale_y

        # filter by size and visibility
        if w < min_box_size or h < min_box_size:
            continue
        # if entirely out of frame, skip
        if (x2 * scale_x) < 0 or (x1 * scale_x) > img_width or (y2 * scale_y) < 0 or (y1 * scale_y) > img_height:
            continue

        name = instance_names.get(track_id, f"kart_{track_id}")

        kart_list.append(
            {
                "instance_id": int(track_id),
                "kart_name": str(name),
                "center": (float(cx), float(cy)),
                "bbox": (float(x1 * scale_x), float(y1 * scale_y), float(x2 * scale_x), float(y2 * scale_y)),
            }
        )

    # identify ego kart:
    ego_index = None
    # prefer track_id == 0 (common convention in your draw function)
    for idx, k in enumerate(kart_list):
        if k["instance_id"] == 0:
            ego_index = idx
            break

    # otherwise choose kart closest to image center
    if ego_index is None and kart_list:
        img_cx = img_width / 2.0
        img_cy = img_height / 2.0
        dists = [((k["center"][0] - img_cx) ** 2 + (k["center"][1] - img_cy) ** 2) for k in kart_list]
        ego_index = int(np.argmin(dists))

    # add is_center_kart flag
    for idx, k in enumerate(kart_list):
        k["is_center_kart"] = (idx == ego_index)

    return kart_list


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string (fallback 'unknown' if not found)
    """
    with open(info_path) as f:
        info = json.load(f)

    # Try common keys
    for key in ("track_name", "track", "map", "level"):
        if key in info and isinstance(info[key], str):
            return info[key]

    # Some files might nest track info
    if "meta" in info and isinstance(info["meta"], dict):
        for key in ("track_name", "track", "map", "level", "name"):
            if key in info["meta"] and isinstance(info["meta"][key], str):
                return info["meta"][key]

    # fallback: maybe a filename is specified
    for key in ("track_file", "track_path"):
        if key in info and isinstance(info[key], str):
            # strip path and extension
            return Path(info[key]).stem

    return "unknown"


def _relative_position(ego_center, other_center, x_thresh=5.0, y_thresh=5.0):
    """
    Return (left/right/center_x, front/behind/center_y) relative labels.
    x_thresh and y_thresh are pixel thresholds for classifying as 'aligned' in that axis.
    """
    ex, ey = ego_center
    ox, oy = other_center
    dx = ox - ex
    dy = oy - ey

    # left/right: other x < ego x => left
    if abs(dx) <= x_thresh:
        lr = "aligned"
    elif dx < 0:
        lr = "left"
    else:
        lr = "right"

    # front/behind: other y < ego y => front (smaller y is towards top)
    if abs(dy) <= y_thresh:
        fb = "aligned"
    elif oy < ey:
        fb = "front"
    else:
        fb = "behind"

    # combined
    if lr == "aligned" and fb == "aligned":
        combined = "same position"
    elif lr == "aligned":
        combined = fb
    elif fb == "aligned":
        combined = lr
    else:
        combined = f"{fb}-{lr}"  # e.g. front-left

    return lr, fb, combined


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    qa = []

    kart_list = extract_kart_objects(info_path, view_index, img_width=img_width, img_height=img_height)

    # find ego
    ego = None
    for k in kart_list:
        if k.get("is_center_kart", False):
            ego = k
            break

    # 1. Ego car question
    if ego is not None:
        q = "What kart is the ego car?"
        a = ego["kart_name"]
        qa.append({"question": q, "answer": a})

    # 2. Total karts question
    total = len(kart_list)
    qa.append({"question": "How many karts are there in the scenario?", "answer": str(total)})

    # 3. Track information question
    track_name = extract_track_info(info_path)
    qa.append({"question": "What track is this?", "answer": track_name})

    # if no ego or only ego, skip relative questions
    if ego is None:
        return qa

    # prepare counters for counting questions
    left_count = 0
    right_count = 0
    front_count = 0
    behind_count = 0

    # thresholds in pixels relative to img size
    x_thresh = max(3.0, img_width * 0.03)
    y_thresh = max(3.0, img_height * 0.03)

    for k in kart_list:
        if k["instance_id"] == ego["instance_id"]:
            continue

        lr, fb, combined = _relative_position(ego["center"], k["center"], x_thresh=x_thresh, y_thresh=y_thresh)

        # increment counters
        if lr == "left":
            left_count += 1
        elif lr == "right":
            right_count += 1
        if fb == "front":
            front_count += 1
        elif fb == "behind":
            behind_count += 1

        # 4. Relative position questions for each kart
        # Is {kart_name} to the left or right of the ego car?
        qa.append(
            {
                "question": f"Is {k['kart_name']} to the left or right of the ego car?",
                "answer": lr if lr != "aligned" else "aligned",
            }
        )
        # Is {kart_name} in front of or behind the ego car?
        qa.append(
            {
                "question": f"Is {k['kart_name']} in front of or behind the ego car?",
                "answer": fb if fb != "aligned" else "aligned",
            }
        )
        # Where is {kart_name} relative to the ego car?
        qa.append({"question": f"Where is {k['kart_name']} relative to the ego car?", "answer": combined})

    # 5. Counting questions
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

def generate_all_qa_pairs(data_dir: str, output_file: str, img_width: int = 150, img_height: int = 100):
    """
    Generate QA pairs for all info.json files in a directory and save them to a JSON file.

    Args:
        data_dir: Directory containing *_info.json files
        output_file: Path to save QA pairs JSON
        img_width: Width for kart center calculations
        img_height: Height for kart center calculations
    """
    data_dir = Path(data_dir)
    all_qa_pairs = []

    info_files = list(data_dir.glob("*_info.json"))
    if not info_files:
        print(f"No info.json files found in {data_dir}")
        return

    for info_file in info_files:
        with open(info_file) as f:
            info = json.load(f)

        num_views = len(info.get("detections", []))
        base_name = info_file.stem.replace("_info", "")

        for view_index in range(num_views):
            image_file_candidates = list(data_dir.glob(f"{base_name}_{view_index:02d}_im.*"))
            if not image_file_candidates:
                continue
            image_file = str(image_file_candidates[0])

            # visualize detections
            annotated_image = draw_detections(image_file, str(info_file))
            plt.figure(figsize=(12, 8))
            plt.imshow(annotated_image)
            plt.axis("off")
            plt.title(f"{base_name} view {view_index}")
            plt.show()

            # generate QA pairs
            qa_pairs = generate_qa_pairs(str(info_file), view_index, img_width=img_width, img_height=img_height)

            # add image filename to each QA pair
            for qa in qa_pairs:
                qa["image_file"] = Path(image_file).as_posix()

            all_qa_pairs.extend(qa_pairs)

    # write to output JSON
    with open(output_file, "w") as f:
        json.dump(all_qa_pairs, f, indent=2)
    print(f"Saved {len(all_qa_pairs)} QA pairs to {output_file}")


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_qa_pairs})
    fire.Fire({"generate_all": generate_all_qa_pairs})


if __name__ == "__main__":
    main()
