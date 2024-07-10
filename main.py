import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, FileResponse
import io
import os
from typing import List
import uuid
import aiohttp
from io import BytesIO
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
from generator import new_palm_masking
from bg_changer import change_bg, perfect_change_bg
from cloth_segmentation import cloth_masking

app = FastAPI()

current_directory = os.path.dirname(__file__)

UPLOAD_DIR = f"{current_directory}/uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

class Base64Image(BaseModel):
    base64_string: str

async def fetch_image_from_url(url: str) -> io.BytesIO:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail="Invalid URL")
            return BytesIO(await response.read())
        
def decode_base64_image(base64_string: str) -> BytesIO:
    image_data = base64.b64decode(base64_string)
    return BytesIO(image_data)
        
@app.post("/image-to-base64/")
async def image_to_base64(file: UploadFile = File(None), url: str = Form(None)):
    if file is None and url is None:
        raise HTTPException(status_code=400, detail="No image file or URL provided")

    if file:
        contents = await file.read()
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_path, "wb") as f:
            f.write(contents)
    else:
        image_io = await fetch_image_from_url(url)
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_path, "wb") as f:
            f.write(image_io.getvalue())

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    os.remove(image_path)
    
    return encoded_string

@app.post("/base64-to-image/")
async def base64_to_image(base64_image: Base64Image):
    try:
        base64_string = base64_image.base64_string
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        image.save(image_path)
        
        return StreamingResponse(open(image_path, "rb"), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{e}")


@app.post("/generate-palm-mask/")
async def palm_mask_endpoint(base64_image: Base64Image):
    try:
        image_io = decode_base64_image(base64_image.base64_string)
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_path, "wb") as f:
            f.write(image_io.getvalue())
        
        palm_mask_path = new_palm_masking(image_path)

        with open(palm_mask_path, "rb") as mask_file:
            mask_base64 = base64.b64encode(mask_file.read()).decode('utf-8')

        os.remove(image_path)
        os.remove(palm_mask_path)

        return mask_base64
    except Exception as e:
        os.remove(image_path)
        raise HTTPException(status_code=400, detail=f"{e}")


@app.post("/change-bg/")
async def bg_change_endpoint(images: List[Base64Image]):
    try:
        if len(images) != 2:
            raise HTTPException(status_code=400, detail="Two images are required")

        # Decode and save the original image
        org_image_io = decode_base64_image(images[0].base64_string)
        org_image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(org_image_path, "wb") as f:
            f.write(org_image_io.getvalue())

        # Decode and save the background image
        bg_image_io = decode_base64_image(images[1].base64_string)
        bg_image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(bg_image_path, "wb") as f:
            f.write(bg_image_io.getvalue())

        # Change the background
        changed_bg_path = change_bg(org_image_path, bg_image_path)

        # Encode the result to base64
        with open(changed_bg_path, "rb") as mask_file:
            changed_bg_base64 = base64.b64encode(mask_file.read()).decode('utf-8')

        # Cleanup
        os.remove(org_image_path)
        os.remove(bg_image_path)
        os.remove(changed_bg_path)

        return changed_bg_base64
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{e}")


@app.post("/generate-cloth-mask/")
async def cloth_mask_endpoint(base64_image: Base64Image):
    try:
        image_io = decode_base64_image(base64_image.base64_string)
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_path, "wb") as f:
            f.write(image_io.getvalue())
        
        cloth_mask_path = cloth_masking(image_path)

        with open(cloth_mask_path, "rb") as mask_file:
            mask_base64 = base64.b64encode(mask_file.read()).decode('utf-8')

        os.remove(image_path)
        os.remove(cloth_mask_path)

        return mask_base64
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{e}")


@app.post("/perfect-change-bg/")
async def perfect_bg_change_endpoint(images: List[Base64Image]):
    try:
        if len(images) != 2:
            raise HTTPException(status_code=400, detail="Two images are required")

        # Decode and save the original image
        org_image_io = decode_base64_image(images[0].base64_string)
        org_image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(org_image_path, "wb") as f:
            f.write(org_image_io.getvalue())

        # Decode and save the background image
        bg_image_io = decode_base64_image(images[1].base64_string)
        bg_image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(bg_image_path, "wb") as f:
            f.write(bg_image_io.getvalue())

        # Change the background
        changed_bg_path = perfect_change_bg(org_image_path, bg_image_path)

        # Encode the result to base64
        with open(changed_bg_path, "rb") as mask_file:
            changed_bg_base64 = base64.b64encode(mask_file.read()).decode('utf-8')

        # Cleanup
        os.remove(org_image_path)
        os.remove(bg_image_path)
        os.remove(changed_bg_path)

        return changed_bg_base64
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
