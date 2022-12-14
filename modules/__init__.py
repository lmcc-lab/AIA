import os
from PIL import Image, ImageDraw
import cv2
import numpy as np
from IPython.display import HTML
from base64 import b64encode
from subprocess import Popen
import os

import torch
from torch import autocast
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from huggingface_hub import notebook_login

save_path = 'results'
cur_dir = os.listdir()

if save_path not in cur_dir:
    os.makedirs('./'+save_path+'/')


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    raise EnvironmentError("CPU isn't enabled with pytorch")

print("Checking downloads are available")
pipe = StableDiffusionPipeline.from_pretrained(
    'CompVis/stable-diffusion-v1-4', revision='fp16',
    torch_dtype=torch.float16, use_auth_token=True)
pipe = pipe.to(device)
