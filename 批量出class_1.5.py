from compel import Compel, ReturnedEmbeddingsType
from diffusers import StableDiffusionPipeline
import torch
import os

# 让用户输入txt文件所在的目录和图片保存的目录
text_files_directory = input("请输入包含prompt的文本文件所在的目录路径:")
output_images_directory = input("请输入生成图片保存的目录路径:")

# 初始化text-to-image管道
pipeline = StableDiffusionPipeline.from_single_file(
    "/mnt/w/model/checkpoints/白城主机甲万象_v1.0.safetensors",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipeline.to("cuda")
pipeline.enable_xformers_memory_efficient_attention()

compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)

# 遍历文本文件目录中的所有文本文件，包括子目录
for root, dirs, files in os.walk(text_files_directory):
    for filename in files:
        if filename.endswith(".txt"):
            # 构建完整的文本文件路径
            file_path = os.path.join(root, filename)
            
            # 读取文件中的prompt
            with open(file_path, 'r') as file:
                prompt = file.read().strip()
                conditioning = compel.build_conditioning_tensor(prompt)

            # 根据prompt生成图片
            generated_images = pipeline(prompt_embeds=conditioning, num_inference_steps=40).images
            
            # 创建与原始文本文件相同结构的目标目录
            relative_dir = os.path.relpath(root, start=text_files_directory)
            target_dir = os.path.join(output_images_directory, relative_dir)
            os.makedirs(target_dir, exist_ok=True)  # 确保目标目录存在
            
            # 保存图片到目标目录
            basename_without_ext = os.path.splitext(filename)[0]
            output_image_path = os.path.join(target_dir, f"{basename_without_ext}.png")
            generated_images[0].save(output_image_path)

            print(f"Generated image saved to {output_image_path}")