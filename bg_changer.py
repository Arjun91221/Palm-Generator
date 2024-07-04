import os
import uuid
from PIL import Image#, ImageFile
from rembg.bg import remove
import numpy as np
from matplotlib import pyplot as plt

current_directory = os.path.dirname(__file__)


def change_bg(org_img, bg_img):
    
    org_img = Image.open(org_img)
    bg_img = Image.open(bg_img)

    mask = remove(org_img, only_mask=True)
    plt.imshow(np.asarray(mask).astype(np.uint8))

    bg_img = bg_img.resize(org_img.size, Image.LANCZOS)
    result = Image.composite(org_img, bg_img, mask)
    result.convert("RGB")

    output_path = f'{current_directory}/uploads/{str(uuid.uuid4())}.png'
    result.save(output_path)

    return output_path

    