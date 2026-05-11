import torch
from diffusers import QwenImageEditInpaintPipeline
from diffusers.utils import load_image

pipe = QwenImageEditInpaintPipeline.from_pretrained("Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

# prompt = 'Fill in the  mirror which corresponds to the mask. The prompt describing the image is: A standing mirror reflects a bed with a dotted cover in a cozy bedroom. Above the bed is a brown rattan headboard and a window with dark gray aluminum.'
prompt = 'Fill in the remaining regions in the mirror which are the regions corresponding to the mask. The prompt describing the image is: A standing mirror reflects a bed with a dotted cover in a cozy bedroom. Above the bed is a brown rattan headboard and a window with dark gray aluminum.'
# source = load_image("/home/ofek_basson/Fill-My-Mirror/data/real_images/images/0.png")
# mask = load_image("/home/ofek_basson/Fill-My-Mirror/data/real_images/masks/0.png")
source = load_image("temp_outputs/projected_image.png")
mask = load_image("temp_outputs/geometry_constraint_mask.png")
image = pipe(
    prompt=prompt,
    negative_prompt=" ",
    image=source,
    mask_image=mask,
    strength=1.0,
    num_inference_steps=30,
).images[0]

image.save("outputs/geometry_qwen_with_qwen_normal.png")
print("Saved to outputs/geometry_qwen_with_qwen_normal.png")
