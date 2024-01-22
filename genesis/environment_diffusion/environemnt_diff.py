import torch
from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler,
)



def get_pip(pretrained , lora ):
    torch.set_default_device('cpu')
    pipe = StableDiffusionPipeline.from_pretrained(pretrained, torch_dtype=torch.float16)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to('cuda')
    torch.set_default_device('cuda')
    pipe.load_lora_weights(lora)
    pipe.fuse_lora(lora_scale=0.9)
    return pipe

def gen_image(prompt, pretrained_model, lora, negative_prompt="", steps = 1000):
    pipe = get_pip(pretrained_model,lora)
    image = pipe(
        prompt, 
        negative_prompt=negative_prompt, 
        width=512,
        height=512,
        guidance_scale=12,
        num_inference_steps=50
    ).images[0]
    return image
