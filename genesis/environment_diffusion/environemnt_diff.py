import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

class EnvironmentDiffusion:
    def __init__(self, lora_path):
        BASE_MODEL = "darkstorm2150/Protogen_x5.3_Official_Release"

        # 初始阶段使用 CPU 减少显存占用
        torch.set_default_device('cpu')
        self.pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            safety_checker=None
        )

        # 替换 scheduler（可选）
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

        # 加载 LoRA
        self.pipe.load_lora_weights(lora_path)

        # 移动到 GPU
        self.pipe.to('cuda')
        torch.set_default_device('cuda')

    def generate(self, prompt, negative_prompt="", steps=25, width=512, height=512, guidance_scale=12):
        result = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=steps
        )
        return result.images[0]
    
    def blend_pirs(self, image1, image2):
        image1 = image1.resize(512,512)