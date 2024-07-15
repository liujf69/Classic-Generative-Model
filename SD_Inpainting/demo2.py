import torch
import torchvision
from PIL import Image
from diffusers import StableDiffusionInpaintPipelineLegacy

if __name__ == "__main__":
    # load inpainting pipeline
    model_id = "runwayml/stable-diffusion-v1-5"
    # model_id = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-waimai-aigc/liujinfu/All_test/test0714/huggingface.co/runwayml/stable-diffusion-v1-5" # local path
    pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(model_id, torch_dtype = torch.float16).to("cuda")

    # load input image and input mask
    input_image = Image.open("./images/overture-creations-5sI6fQgYIuo.png").resize((512, 512))
    input_mask = Image.open("./images/overture-creations-5sI6fQgYIuo_mask.png").resize((512, 512))

    # run inference
    prompt = ["a mecha robot sitting on a bench", "a cat sitting on a bench"]
    generator = torch.Generator("cuda").manual_seed(0)
    with torch.autocast("cuda"):
        images = pipe(
            prompt = prompt,
            image = input_image,
            mask_image = input_mask,
            num_inference_steps = 50,
            strength = 0.75,
            guidance_scale = 7.5,
            num_images_per_prompt = 1,
            generator = generator
        ).images

    # 转成pillow
    for idx, image in enumerate(images):
        image.save("./outputs/output_{:d}.png".format(idx))
    print("All Done!")