import torch
from diffusers import StableDiffusionPipeline


pipe = StableDiffusionPipeline.from_pretrained(
   "CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=True
).to("cuda")

with torch.inference_mode(), torch.autocast("cuda"):
   image = pipe("a small cat", width=1280, height=1280)

