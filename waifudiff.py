from diffusers import DiffusionPipeline
from PIL import Image
import numpy as np
import torch

# Load Waifu Diffusion with FP16 and CUDA
pipe = DiffusionPipeline.from_pretrained(
    "hakurei/waifu-diffusion",
    torch_dtype=torch.float16
).to("cuda")

# Enable memory-efficient attention (Optimized for RTX 4090)
pipe.enable_xformers_memory_efficient_attention()

# Base prompt for character
base_prompt = (
    "1girl, aqua eyes, baseball cap, blonde hair, short hair, simple background, "
    "yellow shirt, hoop earrings, jewelry, upper body, anime style, "
    "detailed, highly detailed face, masterpiece, best quality"
)

# Negative prompt to prevent distortions
negative_prompt = (
    "bad anatomy, extra limbs, multiple faces, low quality, blurry, mutated, "
    "deformed, wrong proportions, watermark, text, ugly, out of focus"
)

def is_black_image(image, threshold=10):
    """Check if the image is mostly black by analyzing pixel values."""
    img_array = np.array(image)
    return np.all(img_array < threshold)  # If all pixel values are very low

# Generate 15 images with a gradual change in looking direction
for i in range(1, 16):  
    generator = torch.Generator(device="cuda").manual_seed(12345)

    looking_direction = f"looking to the right {i*10} degrees"
    prompt = base_prompt + ", " + looking_direction  

    # Generate the image
    image = pipe(prompt, 
                 negative_prompt=negative_prompt, 
                 guidance_scale=7.5,  
                 generator=generator).images[0]

    # Check if image is mostly black
    if is_black_image(image):
        print(f"⚠️ Skipping black image at iteration {i}")
        continue  # Skip saving and move to the next iteration

    # Save only valid images
    filename = f"consistent_waifu_look_{i}.png"
    image.save(filename)
    print(f"✅ Saved: {filename}")

print("🎉 15 Images with Gradual Looking Direction Change Generated Successfully!")
