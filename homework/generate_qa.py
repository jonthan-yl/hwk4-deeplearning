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

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    # Load info.json
    with open(info_path) as f:
        info = json.load(f)

    # Get detections for the requested view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        frame_detections = []

    # Scaling factors from original coordinates to requested sized image
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Collect kart objects
    karts = []
    for det in frame_detections:
        # Each detection entry: [class_id, track_id, x1, y1, x2, y2]
        class_id = int(det[0])
        track_id = int(det[1])
        x1 = float(det[2])
        y1 = float(det[3])
        x2 = float(det[4])
        y2 = float(det[5])

        if class_id != 1:
            # Only consider class 1 (Kart)
            continue

        # Compute bounding box center and size
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        # Ignore tiny boxes or invalid ones
        if width < min_box_size or height < min_box_size:
            continue

        # Filter out boxes that are completely outside the original image
        if x2 < 0 or x1 > ORIGINAL_WIDTH or y2 < 0 or y1 > ORIGINAL_HEIGHT:
            continue

        # Scale centers to requested image size
        center_x_scaled = center_x * scale_x
        center_y_scaled = center_y * scale_y

        # Kart name (if provided in info['karts'])
        kart_name = None
        if "karts" in info and isinstance(info["karts"], list):
            if 0 <= track_id < len(info["karts"]):
                kart_name = info["karts"][track_id]
        if kart_name is None:
            kart_name = f"kart_{track_id}"

        karts.append(
            {
                "instance_id": track_id,
                "kart_name": kart_name,
                "center": (center_x_scaled, center_y_scaled),
                "bbox": (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y),
            }
        )

    # Identify the center kart: either the one with track_id==0 (ego), otherwise the kart closest to image center
    ego_track_id = None
    if "karts" in info and len(info.get("karts", [])) > 0:
        # If there's a known ego index (0), prefer that
        ego_track_id = 0

    # Determine center of the image (in scaled coordinates)
    center_image_x = img_width / 2.0
    center_image_y = img_height / 2.0

    # If we couldn't find an ego by track_id, fall back to nearest to image center
    if ego_track_id is None:
        # compute distances and select nearest
        if len(karts) > 0:
            distances = [((k["center"][0] - center_image_x) ** 2 + (k["center"][1] - center_image_y) ** 2) for k in karts]
            min_idx = int(np.argmin(distances))
            ego_track_id = karts[min_idx]["instance_id"]

    # Set is_center_kart flag
    for k in karts:
        k["is_center_kart"] = k["instance_id"] == ego_track_id

    return karts


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    with open(info_path) as f:
        info = json.load(f)

    # The dataset stores the track name under 'track'
    track_name = info.get("track", None)
    if track_name is None:
        # Fallback to reading from file name if possible
        info_name = Path(info_path).stem
        track_name = info_name
    return track_name


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
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    # Parse track name and karts
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    # Find ego kart entry
    ego = None
    for k in karts:
        if k.get("is_center_kart"):
            ego = k
            break

    # Build image_file path, matching existing sample format: <split>/<file>_XX_im.jpg
    info_path_obj = Path(info_path)
    base_name = info_path_obj.stem.replace("_info", "")
    image_filename = list(info_path_obj.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))
    image_file = None
    if image_filename:
        image_file = f"{info_path_obj.parent.name}/{image_filename[0].name}"
    else:
        # Fall back to just using base name
        image_file = f"{info_path_obj.parent.name}/{base_name}_{view_index:02d}_im.jpg"

    qa_pairs = []

    # 1) Ego car question
    if ego is not None:
        qa_pairs.append(
            {
                "question": "What kart is the ego car?",
                "answer": str(ego.get("kart_name", f"kart_{ego['instance_id']}")),
                "image_file": image_file,
            }
        )

    # 2) Total karts question
    qa_pairs.append(
        {"question": "How many karts are there in the scenario?", "answer": str(len(karts)), "image_file": image_file}
    )

    # 3) Track name question
    if track_name is not None:
        qa_pairs.append({"question": "What track is this?", "answer": str(track_name), "image_file": image_file})

    # If no ego, we can't do relative questions properly; but still provide counting
    if ego is None:
        return qa_pairs

    ego_x, ego_y = ego["center"]

    # Prepare counts and per-kart relative questions
    left_count = right_count = front_count = back_count = 0

    for k in karts:
        if k["instance_id"] == ego["instance_id"]:
            continue

        kx, ky = k["center"]

        # Determine left/right
        lr = "right" if kx > ego_x else "left"
        if lr == "right":
            right_count += 1
        else:
            left_count += 1

        # Determine front/back (y smaller means more 'front' in image, assuming top-left origin)
        fb = "front" if ky < ego_y else "back"
        if fb == "front":
            front_count += 1
        else:
            back_count += 1

        # Add relative Q: front/back, left/right, combined
        qa_pairs.append({"question": f"Is {k['kart_name']} in front of or behind the ego car?", "answer": fb, "image_file": image_file})
        qa_pairs.append({"question": f"Is {k['kart_name']} to the left or right of the ego car?", "answer": lr, "image_file": image_file})
        # combined descriptive answer: 'front and left' or 'back' etc.
        combined = fb
        if (lr != "left" and lr != "right"):
            # Should never happen; skip
            pass
        else:
            combined = fb
            if lr:
                combined = f"{fb} and {lr}"

        qa_pairs.append({"question": f"Where is {k['kart_name']} relative to the ego car?", "answer": combined, "image_file": image_file})

    # 5) Counting questions
    qa_pairs.append({"question": "How many karts are to the left of the ego car?", "answer": str(left_count), "image_file": image_file})
    qa_pairs.append({"question": "How many karts are to the right of the ego car?", "answer": str(right_count), "image_file": image_file})
    qa_pairs.append({"question": "How many karts are in front of the ego car?", "answer": str(front_count), "image_file": image_file})
    qa_pairs.append({"question": "How many karts are behind the ego car?", "answer": str(back_count), "image_file": image_file})

    return qa_pairs


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


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_qa_pairs})


if __name__ == "__main__":
    main()
