import cv2
import json
import os
import numpy as np
import sys
from ultralytics import YOLO


def _detect_players(model, frame):
    """Run YOLO on one frame and return its player detections."""
    yolo_results = model(frame, verbose=False)
    detections = []

    for r in yolo_results:
        for box in r.boxes:
            cls_id = int(box.cls.cpu().item())
            conf = float(box.conf.cpu().item())
            label = model.names[cls_id]

            if label.lower() != "player":
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            width = x2 - x1
            height = y2 - y1
            center_x = x1 + width / 2
            center_y = y1 + height / 2

            detections.append({
                "class": label,
                "class_id": cls_id,
                "confidence": conf,
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "width": float(width),
                    "height": float(height),
                    "center_x": float(center_x),
                    "center_y": float(center_y)
                }
            })
    return detections


def _load_snap_frame(snap_detection_path):
    with open(snap_detection_path, 'r') as f:
        snaps = json.load(f).get('snaps', [])
    if not snaps:
        raise ValueError(f"No snap found in {snap_detection_path}")
    return snaps[0].get('frame')


def playerDetection(video_path, model_path="yolo_models/bestPlayerDetectorM.pt", output_path="cache/playerDetection/playerDetection.json", stride=1, snap_detection_path=None):
    """
    Detect players in a video file

    Args:
        video_path: Path to input video file
        model_path: Path to YOLO model weights
        output_path: Path to output JSON file
        stride: Only detect every Nth frame (1 = every frame). A coarse pass is
            enough for snap detection.
        snap_detection_path: If given, detect only the snap frame and merge it
            into the existing output file (run after snap detection).

    Returns:
        Dictionary with detection results
    """
    # Load YOLO model
    model = YOLO(model_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Processing video: {video_path}")
    print(f"FPS: {fps}, Total frames: {total_frames}")
    print(f"Resolution: {width}x{height}")

    # Initialize results structure
    results = {
        "video_info": {
            "path": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height
        },
        "frames": []
    }

    if snap_detection_path:
        # Snap-frame pass: keep the coarse frames already on disk, add the snap frame.
        snap_frame = _load_snap_frame(snap_detection_path)
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                results = json.load(f)

        # Seek the same way positionDetection.py does so both see the identical frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, snap_frame)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read snap frame {snap_frame}")

        frames = [fr for fr in results.get("frames", []) if fr.get("frame_number") != snap_frame]
        frames.append({
            "frame_number": snap_frame,
            "timestamp": snap_frame / fps,
            "detections": _detect_players(model, frame)
        })
        frames.sort(key=lambda fr: fr["frame_number"])
        results["frames"] = frames
        print(f"Processed snap frame {snap_frame}")
    else:
        stride = max(1, int(stride))
        frame_number = 0

        while True:
            if frame_number % stride != 0:
                # Skipped frame: advance without decoding into an image
                if not cap.grab():
                    break
                frame_number += 1
                continue

            ret, frame = cap.read()
            if not ret:
                break

            # Add detection data to results
            results["frames"].append({
                "frame_number": frame_number,
                "timestamp": frame_number / fps,
                "detections": _detect_players(model, frame)
            })

            if frame_number % 5 == 0:
                print(f"Processed frame {frame_number}/{total_frames}", flush=True)

            frame_number += 1

    cap.release()

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Player detection complete. Results saved to: {output_path}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Player Detection Module (Video)')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='cache/playerDetection/playerDetection.json', help='Path to output JSON file')
    parser.add_argument('--model', type=str, default='yolo_models/bestPlayerDetectorM.pt', help='Path to YOLO model weights')
    parser.add_argument('--stride', type=int, default=1, help='Only detect every Nth frame (default 1 = every frame)')
    parser.add_argument('--snap-detection', type=str, default=None, help='Snap detection JSON: detect only the snap frame and merge it into --output')
    args = parser.parse_args()

    try:
        results = playerDetection(args.video, args.model, args.output, args.stride, args.snap_detection)
        total_detections = sum(len(frame['detections']) for frame in results['frames'])
        print(f"Detected {total_detections} player objects across {len(results['frames'])} frames")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
