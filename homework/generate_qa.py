import json
import glob
import os
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
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        try:
            frame_id = int(parts[0], 16) if len(parts[0]) > 0 else 0 # Convert hex to decimal
            view_index = int(parts[1])
            return frame_id, view_index
        except ValueError:
            pass
    return 0, 0


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.
    """
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    img_width, img_height = pil_image.size
    draw = ImageDraw.Draw(pil_image)

    with open(info_path) as f:
        info = json.load(f)

    _, view_index = extract_frame_info(image_path)

    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        return np.array(pil_image)

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue
        
        # Check bounds
        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        if track_id == 0:
            color = (255, 0, 0) # Red for Ego
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file.
    """
    with open(info_path, 'r') as f:
        data = json.load(f)

    if view_index >= len(data["detections"]):
        return []

    detections = data["detections"][view_index]
    kart_names = data.get("kart_names", [])
    
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    visible_karts = []

    for det in detections:
        class_id, track_id, x1, y1, x2, y2 = det
        
        # We only care about Karts (Class ID 1)
        if int(class_id) != 1:
            continue

        # Scale coordinates
        x1 = x1 * scale_x
        y1 = y1 * scale_y
        x2 = x2 * scale_x
        y2 = y2 * scale_y

        # 1. Filter out-of-bounds or too small
        if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
            continue
        if x2 < 0 or x1 > img_width or y2 < 0 or y1 > img_height:
            continue

        # 2. Get Center
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # 3. Get Name
        name = "unknown"
        if 0 <= int(track_id) < len(kart_names):
            name = kart_names[int(track_id)]

        visible_karts.append({
            "track_id": int(track_id),
            "name": name,
            "center": (cx, cy),
            "bbox": (x1, y1, x2, y2)
        })

    # Determine center kart (closest to image center X)
    image_center_x = img_width / 2
    if visible_karts:
        # Sort by distance to center x
        visible_karts.sort(key=lambda k: abs(k["center"][0] - image_center_x))
        visible_karts[0]["is_center_kart"] = True
        for k in visible_karts[1:]:
            k["is_center_kart"] = False
            
    return visible_karts


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.
    """
    with open(info_path, 'r') as f:
        data = json.load(f)
    return data.get("track", "unknown")


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.
    """
    qa_pairs = []
    
    # Load data
    with open(info_path, 'r') as f:
        data = json.load(f)
        
    kart_names = data.get("kart_names", [])
    track_name = extract_track_info(info_path)
    
    # Get visible objects
    visible_karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    
    # Helper to find Ego
    ego_kart = next((k for k in visible_karts if k["track_id"] == 0), None)
    
    # If Ego is not visible, we assume the camera IS the ego position (bottom center)
    # for spatial comparison purposes.
    if ego_kart:
        ego_x, ego_y = ego_kart["center"]
    else:
        ego_x, ego_y = img_width / 2, img_height  # Bottom center

    # ---------------------------------------------------------
    # 1. Ego car question
    # ---------------------------------------------------------
    if kart_names and len(kart_names) > 0:
        ego_name = kart_names[0] # track_id 0 is always ego
        qa_pairs.append({
            "question": "What kart is the ego car?",
            "answer": ego_name
        })

    # ---------------------------------------------------------
    # 2. Total karts question
    # ---------------------------------------------------------
    # Count how many karts are in the scenario (visible in this frame)
    # Note: Usually implies "other" karts, but prompt demo says "4", 
    # so we count all visible bounding boxes.
    count = len(visible_karts)
    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(count)
    })

    # ---------------------------------------------------------
    # 3. Track information questions
    # ---------------------------------------------------------
    qa_pairs.append({
        "question": "What track is this?",
        "answer": track_name
    })

    # ---------------------------------------------------------
    # 4. Relative position & 5. Counting relative
    # ---------------------------------------------------------
    
    left_count = 0
    right_count = 0
    front_count = 0
    behind_count = 0

    for kart in visible_karts:
        if kart["track_id"] == 0:
            continue # Skip comparing ego to itself

        k_name = kart["name"]
        kx, ky = kart["center"]

        # Left/Right Logic
        # If kart x is less than ego x, it's left.
        h_pos = ""
        if kx < ego_x:
            h_pos = "left"
            left_count += 1
            qa_pairs.append({
                "question": f"Is {k_name} to the left or right of the ego car?",
                "answer": "left"
            })
        else:
            h_pos = "right"
            right_count += 1
            qa_pairs.append({
                "question": f"Is {k_name} to the left or right of the ego car?",
                "answer": "right"
            })

        # Front/Behind Logic (2D Approximation)
        # In images: Lower Y value (top of screen) = Further away (Front)
        # Higher Y value (bottom of screen) = Closer (Behind, assuming ego is driving forward)
        v_pos = ""
        if ky < ego_y:
            v_pos = "front"
            front_count += 1
            qa_pairs.append({
                "question": f"Is {k_name} in front of or behind the ego car?",
                "answer": "front"
            })
        else:
            v_pos = "behind"
            behind_count += 1
            qa_pairs.append({
                "question": f"Is {k_name} in front of or behind the ego car?",
                "answer": "behind"
            })

        # General relative question
        # "Where is {kart_name} relative to the ego car?" -> "front left", "behind right", etc.
        qa_pairs.append({
            "question": f"Where is {k_name} relative to the ego car?",
            "answer": f"{v_pos} {h_pos}"
        })

    # Counting Questions
    qa_pairs.append({
        "question": "How many karts are to the left of the ego car?",
        "answer": str(left_count)
    })
    qa_pairs.append({
        "question": "How many karts are to the right of the ego car?",
        "answer": str(right_count)
    })
    qa_pairs.append({
        "question": "How many karts are in front of the ego car?",
        "answer": str(front_count)
    })
    qa_pairs.append({
        "question": "How many karts are behind the ego car?",
        "answer": str(behind_count)
    })

    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.
    """
    info_path = Path(info_file)
    # Parse filename to handle different naming conventions if needed
    # Assuming standard: frame_view_im.jpg or similar mapping based on folder structure
    # Standard SuperTux structure often has images in same folder
    
    # Construct expected image filename
    # format: {frame_id:05d}_{view_index:02d}_im.jpg
    # We need to extract frame_id from info file name "00000_info.json" -> 0
    frame_id_str = info_path.stem.split("_")[0] 
    image_name = f"{frame_id_str}_{view_index:02d}_im.jpg"
    image_file = info_path.parent / image_name

    if not image_file.exists():
        print(f"Error: Could not find image file {image_file}")
        return

    # Visualize detections
    try:
        annotated_image = draw_detections(str(image_file), info_file)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_image)
        plt.axis("off")
        plt.title(f"Frame {frame_id_str}, View {view_index}")
        plt.show()
    except Exception as e:
        print(f"Could not draw detections: {e}")

    # Generate QA pairs
    try:
        qa_pairs = generate_qa_pairs(info_file, view_index)
        print("\nQuestion-Answer Pairs:")
        print("-" * 50)
        for qa in qa_pairs:
            print(f"Q: {qa['question']}")
            print(f"A: {qa['answer']}")
        print("-" * 50)
    except Exception as e:
        print(f"Error generating QA: {e}")
        import traceback
        traceback.print_exc()


def generate_dataset(data_dir: str, output_file: str = "train_qa.json"):
    """
    Generate the massive VLM training dataset by iterating over all data.
    
    Args:
        data_dir: Directory containing the data (e.g., 'train')
        output_file: Output JSON file path
    """
    data_path = Path(data_dir)
    all_qa_data = []
    
    # Find all info.json files
    info_files = sorted(list(data_path.glob("*_info.json")))
    print(f"Found {len(info_files)} info files. Processing...")
    
    count = 0
    for info_file in info_files:
        # Each info file might correspond to multiple views/images
        # Usually check how many detections lists exist to know how many views
        try:
            with open(info_file, 'r') as f:
                data = json.load(f)
                num_views = len(data.get("detections", []))
            
            frame_id_str = info_file.stem.split("_")[0]
            
            for view_idx in range(num_views):
                # Check if image exists
                image_name = f"{frame_id_str}_{view_idx:02d}_im.jpg"
                image_path = info_file.parent / image_name
                
                if not image_path.exists():
                    continue
                
                # Generate QA pairs
                pairs = generate_qa_pairs(str(info_file), view_idx)
                
                # Format for VLM training
                # Add image_file path relative to dataset root or absolute
                # The demo format was: {"question":.., "answer":.., "image_file":..}
                
                # Create relative path str roughly matching "train/xxxxx_xx_im.jpg"
                rel_image_path = f"{data_path.name}/{image_name}"
                
                for p in pairs:
                    p["image_file"] = rel_image_path
                    all_qa_data.append(p)
                    
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} files...")
                
        except Exception as e:
            print(f"Skipping {info_file} due to error: {e}")

    print(f"Generation complete. Total QA pairs: {len(all_qa_data)}")
    with open(output_file, 'w') as f:
        json.dump(all_qa_data, f, indent=2)
    print(f"Saved to {output_file}")


def main():
    fire.Fire({
        "check": check_qa_pairs,
        "generate": generate_dataset
    })


if __name__ == "__main__":
    main()  