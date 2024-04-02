import io
import cv2
import base64
import requests
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np

app = FastAPI()
webui_url = "http://54.147.7.5:7860"


def A1111_Payload(
        prompt, control_img,
        negative_prompt='ugly, blurry, noisy, low quality, cartoon, 3D, drawing, painting, sketch, scribble, '
                        'disfigured, disproportional, mannequin, earrings, buck teeth, distortion, nude, naked, nsfw',
        controlnet_module='canny',
        controlnet_model='diffusers_xl_canny_full [2b69fca4]',
        sdxl_model='sd_xl_base_1.0.safetensors [31e35c80fc]',
        sampler='Euler a',
):
    control_img_array = np.frombuffer(control_img, np.uint8)
    control_img = cv2.imdecode(control_img_array, cv2.IMREAD_COLOR)
    retval, bytes = cv2.imencode('.png', control_img)
    control_encoded_image = base64.b64encode(bytes).decode('utf-8')

    # A1111 payload
    payload = {

        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "batch_size": 1,
        "steps": 20,
        "cfg_scale": 7,
        "sd_model_checkpoint": sdxl_model,
        "width": 1024,
        "height": 1024,
        "sampler_name": sampler,
        "n_iter": 1,
        "alwayson_scripts": {
            "ControlNet": {
                "args": [
                    {
                        "batch_images": "",
                        "control_mode": "Balanced",
                        "enabled": True,
                        "guidance_end": 1,
                        "guidance_start": 0,
                        "is_ui": True,
                        "image": {"image": control_encoded_image, "mask": None},
                        "module": controlnet_module,
                        "model": controlnet_model,
                        "pixel_perfect": True,
                        "resize_mode": "Crop and Resize",
                        "threshold_a": 100,
                        "threshold_b": 200,
                        "weight": 1
                    }
                ]
            }
        }
    }

    return payload


@app.post("/txt2img_canny")
async def text_to_img_canny(prompt: str, control_img: UploadFile = File(...)):
    try:
        ctrlnet_img = await control_img.read()
        a1111_payload = A1111_Payload(prompt, ctrlnet_img)
        response = requests.post(url=f'{webui_url}/sdapi/v1/txt2img', json=a1111_payload)
        response.raise_for_status()  # Check for HTTP errors
        r = response.json()
        if 'images' in r:
            image_data = r['images'][0]
            image_data_1 = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(base64.b64decode(image_data.split(",", 1)[0])))
            # image.save(r"/home/ubuntu/aakarsh/webui-server/image1.jpg")
            return StreamingResponse(io.BytesIO(image_data_1), media_type="image/png")
        else:
            return {"error": "Unexpected response format"}
    except Exception as e:
        return {"error": str(e)}

