#!/usr/bin/env python3
"""
Segment image using SAM 2 and Grounding DINO.
Based on: https://github.com/Jianghanxiao/proj-QQTT/blob/main/data_process/segment_util_image.py

Requires SAM2 and Grounding DINO checkpoints in data_process/groundedSAM_checkpoints/
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

import cv2
import torch
import numpy as np
from torchvision.ops import box_convert

# Try to import SAM2 and Grounding DINO
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("[WARNING] SAM2 not installed. Install via: pip install sam-2")

try:
    from groundingdino.util.inference import load_model, load_image, predict
except ImportError:
    print("[WARNING] Grounding DINO not installed.")


# Configuration
SAM2_CHECKPOINT = "data_process/groundedSAM_checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "data_process/groundedSAM_checkpoints/configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "data_process/groundedSAM_checkpoints/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "data_process/groundedSAM_checkpoints/groundingdino_swint_ogc.pth"

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def segment_image(
    img_path: str,
    text_prompt: str,
    output_path: str,
    visualize: bool = False,
    save_mask: bool = True,
) -> Optional[np.ndarray]:
    """
    Segment image using SAM2 and Grounding DINO.
    
    Args:
        img_path: Path to input image
        text_prompt: Text description of object to segment (lowercase, ending with dot)
        output_path: Path to save segmented image with alpha channel
        visualize: Show intermediate results
        save_mask: Save the binary mask as PNG
    
    Returns:
        masks: (N, H, W) binary masks or None if segmentation failed
    """
    print(f"[INFO] Loading image: {img_path}")
    image_source, image = load_image(img_path)
    h, w, _ = image_source.shape
    
    # Initialize SAM2
    print("[INFO] Initializing SAM2...")
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    sam2_predictor.set_image(image_source)
    
    # Initialize Grounding DINO
    print("[INFO] Loading Grounding DINO model...")
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE,
    )
    
    # Ensure text prompt is lowercase and ends with dot
    if not text_prompt.endswith("."):
        text_prompt = text_prompt.lower() + "."
    else:
        text_prompt = text_prompt.lower()
    
    print(f"[INFO] Text prompt: '{text_prompt}'")
    
    # Run Grounding DINO
    print("[INFO] Running Grounding DINO...")
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )
    
    print(f"[INFO] Detected {len(boxes)} objects")
    
    if len(boxes) == 0:
        print("[WARNING] No objects detected. Returning empty mask.")
        return None
    
    # Convert boxes to SAM2 format (xyxy)
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    # Run SAM2
    print("[INFO] Running SAM2 prediction...")
    with torch.autocast(device_type=DEVICE, dtype=torch.float16):
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
    
    # Convert to (N, H, W) format if needed
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    print(f"[INFO] Generated {len(masks)} masks")
    
    # Save result image (overlay mask on RGB with alpha channel)
    raw_img = cv2.imread(img_path)
    output_img = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Use first mask (highest confidence)
    mask_bool = masks[0] > 0
    output_img[mask_bool, :3] = raw_img[mask_bool]
    output_img[:, :, 3] = mask_bool.astype(np.uint8) * 255
    
    cv2.imwrite(output_path, output_img)
    print(f"[INFO] Saved segmented image: {output_path}")
    
    # Optionally save binary mask (for single camera, use standard name)
    if save_mask:
        mask_path = str(Path(output_path).parent / "mask.png")
        mask_uint8 = (masks[0] * 255).astype(np.uint8)
        cv2.imwrite(mask_path, mask_uint8)
        print(f"[INFO] Saved binary mask: {mask_path}")
    
    # Optionally visualize
    if visualize:
        for i, mask in enumerate(masks):
            mask_img = (mask * 255).astype(np.uint8)
            cv2.imshow(f"Mask {i}", mask_img)
        cv2.imshow("Original", raw_img)
        cv2.imshow("Segmented", output_img)
        print("[INFO] Press any key to close visualization windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return masks


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Segment image using SAM2 and Grounding DINO"
    )
    parser.add_argument(
        "--img_path", type=str, default="outputs/color.png", help="Path to input image"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/color_segmented.png",
        help="Path to save segmented image",
    )
    parser.add_argument(
        "--text_prompt", type=str, default="rope.", help="Text description of object"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize results"
    )
    parser.add_argument(
        "--no-mask", action="store_true", help="Don't save binary mask"
    )
    
    args = parser.parse_args()
    
    segment_image(
        img_path=args.img_path,
        text_prompt=args.text_prompt,
        output_path=args.output_path,
        visualize=args.visualize,
        save_mask=not args.no_mask,
    )


if __name__ == "__main__":
    main()
