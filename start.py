from modules import *
from modules.functions import gen_image, image_grid


if __name__ == '__main__':
    print("Checking downloads are available")
    pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', revision='fp16',
        torch_dtype=torch.float16, use_auth_token=True)
    pipe = pipe.to(device)

    while True:
        prompt = input("What would you like to see?: ")
        while True:
            try:
                number_images = input("How many do you want?: ")
                number_images = int(number_images)
                break
            except ValueError as e:
                print("Number of images wasn't an integer. Try again")

        if number_images <= 1:
            image = gen_image(pipe, prompt)
            print(f"Image saved to {save_path}")
        else:
            prompts = [prompt] * number_images
            with autocast(device):
                images = pipe(prompts, num_inference_steps=50)['sample']
            grid = image_grid(images, rows=1, cols=3)



