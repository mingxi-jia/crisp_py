#!/usr/bin/env python3
"""
Visualize the initial position randomization range by overlaying the first frame
of camera 3 from every episode at low alpha onto a single composite image.

Usage:
    python scripts/visualize_init_randomization.py /media/mingxi/T7/XEMB_Experiment/coffee_making/episodes_d4
    python scripts/visualize_init_randomization.py /path/to/episodes --cam 1
    python scripts/visualize_init_randomization.py /path/to/episodes --brightness 1.5 --output out.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def collect_first_frames(episodes_path: Path, cam_idx: int = 3, use_segmented: bool = False):
    """Return a list of (episode_name, image_path) for the first frame of each episode."""
    img_subdir = "segmented_rgb" if use_segmented else "rgb"

    episodes = sorted(
        d for d in episodes_path.iterdir()
        if d.is_dir() and d.name.startswith("episode_")
    )

    if not episodes:
        print(f"No episodes found in {episodes_path}")
        return []

    results = []
    missing = 0
    for ep in episodes:
        cam_dir = ep / f"cam{cam_idx}" / img_subdir
        if not cam_dir.exists():
            print(f"  [skip] {ep.name}: {cam_dir} does not exist")
            missing += 1
            continue

        images = sorted(cam_dir.glob("*.png"))
        if not images:
            print(f"  [skip] {ep.name}: no images in {cam_dir}")
            missing += 1
            continue

        results.append((ep.name, images[0]))

    print(f"Found {len(results)} episodes with cam{cam_idx} frames ({missing} skipped)")
    return results


def build_composite(frame_paths: list[tuple[str, Path]], brightness: float = 1.0) -> np.ndarray:
    """
    Average all frames pixel-wise.

    Static parts of the scene (same across all episodes) appear at full colour.
    Objects at randomised positions each contribute 1/N of the brightness, so
    they appear as faint ghosts — giving an intuitive view of the randomisation range.

    brightness: optional multiplier applied after averaging (default 1.0 = no change).
    """
    imgs: list[np.ndarray] = []
    ref_shape: tuple[int, ...] | None = None

    for _, path in frame_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  [warn] could not read {path}")
            continue
        if ref_shape is None:
            ref_shape = img.shape
        elif img.shape != ref_shape:
            img = cv2.resize(img, (ref_shape[1], ref_shape[0]))
        imgs.append(img.astype(np.float64))

    if not imgs or ref_shape is None:
        raise RuntimeError("No images could be loaded.")

    # Mean across all frames — equivalent to each frame at opacity 1/N
    canvas = np.mean(imgs, axis=0)

    if brightness != 1.0:
        canvas = canvas * brightness

    return np.clip(canvas, 0, 255).astype(np.uint8)


def add_info_overlay(composite: np.ndarray, num_episodes: int, cam_idx: int) -> np.ndarray:
    """Add a small text banner at the top of the composite."""
    banner_h = 50
    banner = np.zeros((banner_h, composite.shape[1], 3), dtype=np.uint8)
    text = f"cam{cam_idx} first-frame overlay  |  {num_episodes} episodes  |  pixel mean"
    cv2.putText(banner, text, (10, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 50), 2, cv2.LINE_AA)
    return np.vstack([banner, composite])


def main():
    parser = argparse.ArgumentParser(
        description="Overlay first frames of a given camera across all episodes to show randomisation range.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("path", type=str,
                        help="Path to episodes root directory (contains episode_* folders)")
    parser.add_argument("--cam", type=int, default=3,
                        help="Camera index to use (default: 3)")
    parser.add_argument("--brightness", type=float, default=1.0,
                        help="Brightness multiplier applied after averaging (default: 1.0)")
    parser.add_argument("--segment", action="store_true",
                        help="Use segmented_rgb instead of rgb")
    parser.add_argument("--output", type=str, default=None,
                        help="Save composite to this file instead of showing a window")
    args = parser.parse_args()

    episodes_path = Path(args.path)
    if not episodes_path.exists():
        print(f"Error: path does not exist: {episodes_path}")
        return

    frame_paths = collect_first_frames(episodes_path, cam_idx=args.cam,
                                       use_segmented=args.segment)
    if not frame_paths:
        return

    n = len(frame_paths)
    print(f"Averaging {n} frames (pixel mean), brightness={args.brightness}")

    composite = build_composite(frame_paths, brightness=args.brightness)
    composite = add_info_overlay(composite, n, args.cam)

    if args.output:
        out_path = Path(args.output)
        cv2.imwrite(str(out_path), composite)
        print(f"Saved composite to {out_path}")
    else:
        win = "Randomisation Range (press any key to close)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        # Scale window to a reasonable size (max 1600 wide)
        h, w = composite.shape[:2]
        scale = min(1.0, 1600 / w)
        cv2.resizeWindow(win, int(w * scale), int(h * scale))
        cv2.imshow(win, composite)
        print("Press any key in the image window to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
