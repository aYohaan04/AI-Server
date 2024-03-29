import io
import cv2
import base64
import requests
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import numpy as np

app = FastAPI()
webui_url = "http://127.0.0.1:7861/"


def A1111_Payload(
        prompt, control_img,
        negative_prompt='ugly, blurry, noisy, low quality, cartoon, 3D, drawing, painting, sketch, scribble, '
                        'disfigured, disproportional, mannequin, earrings, buck teeth, distortion, nude, naked, nsfw',
        controlnet_module='canny',
        controlnet_model='control_v11p_sd15_canny [d14c016b]',
        sdxl_model='sd_xl_base_1.0.safetensors [31e35c80fc]',
        sampler='DPM++ 2M Karras',
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
        "steps": 35,
        "cfg_scale": 7,
        "sd_model_checkpoint": sdxl_model,
        "width": 1024,
        "height": 1024,
        "sampler_name": sampler,
        "sampler_index": sampler,
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "input_image": control_encoded_image,
                        "module": controlnet_module,
                        "model": controlnet_model,
                        "pixel_perfect": True,
                        "resize_mode": 2,
                        "control_mode": 1,
                    }
                ]
            }
        }
    }

    return payload


@app.post("/txt2img_canny")
async def text_to_img_canny(prompt: str, control_img: UploadFile = File(...)):
    ctrlnet_img = await control_img.read()
    a1111_payload = A1111_Payload(prompt, ctrlnet_img)
    response = requests.post(url=f'{webui_url}/sdapi/v1/txt2img', json=a1111_payload)
    r = response.json()
    print("Resuklt:", r)
    print("Prompt: ", prompt)
    result = r['images'][0]
    out_image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
    out_image.save(r"E:\AI_ML_Projects\ArchitectSDXL\AI_Server\test.jpg")
    return result
