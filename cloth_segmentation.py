import os
import uuid
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import torch.nn as nn

current_directory = os.path.dirname(__file__)

def cloth_masking(image_path):
    """
    Generate a binary mask for clothing regions in the input image.

    Args:
    image_path (str): Path to the input image file.

    Returns:
    numpy.ndarray: Binary mask of clothing regions.
    """
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]

    clothes_labels = [1, 4, 5, 6, 7, 8, 16, 17]  # Hat, Upper-clothes, Skirt, Pants, Dress, Belt, Bag, Scarf
    clothes_mask = torch.zeros_like(pred_seg, dtype=torch.bool)
    for label in clothes_labels:
        clothes_mask |= (pred_seg == label)
    clothes_mask_image = clothes_mask.numpy().astype("uint8") * 255

    output_path = f'{current_directory}/uploads/{str(uuid.uuid4())}.png'
    Image.fromarray(clothes_mask_image).save(output_path)

    return output_path
