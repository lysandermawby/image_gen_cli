#!/usr/bin/python
"""Generate images using the Illustrious-XL model"""

# package imports
import torch
from diffusers import StableDiffusionXLPipeline
import argparse
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# local imports
import callback_utils
# from print_utils import print_debug

def check_device():
    """check what device is available"""

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("No CUDA support - running on CPU")
    
    return device


def generate_image(
    pipe,
    prompt,
    negative_prompt,
    num_inference_steps,
    guidance_scale,
    width,
    height,
    device,
    seed,
    save_gif, 
    gif_name
):
    """generating the image using a local diffusion model"""

    # Validation
    if prompt is None or prompt.strip() == "":
        raise ValueError("Prompt cannot be None or empty")
    
    if num_inference_steps is None or not isinstance(num_inference_steps, int) or num_inference_steps < 1:
        raise ValueError(f"num_inference_steps must be >= 1, got {num_inference_steps}")
    
    if guidance_scale is None or not isinstance(guidance_scale, (int, float)) or guidance_scale < 0:
        raise ValueError(f"guidance_scale must be >= 0, got {guidance_scale}")
    
    if width is None or width < 64 or width % 8 != 0:
        raise ValueError(f"width must be >= 64 and divisible by 8, got {width}")
    
    if height is None or height < 64 or height % 8 != 0:
        raise ValueError(f"height must be >= 64 and divisible by 8, got {height}")
    
    # Setup generator for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))

        # Setup callback for progress tracking (NEW API)
    callback_on_step_end = None
    intermediate_images = []
    if save_gif:
        callback_on_step_end, intermediate_images = callback_utils.create_progress_callback()

    print("Starting image generation...")
    
    image = pipe(
        callback_on_step_end_tensor_inputs=["latents"],
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        width=int(width),
        height=int(height),
        generator=generator,
        callback_on_step_end=callback_on_step_end,
        num_images_per_prompt=1
    ).images[0]

    return image, intermediate_images


def save_image(image, image_name):
    """saving the image"""

    # automatically assumes that the saved image should be a png
    image_path = Path(f"images/{image_name}.png")  
    os.makedirs("images", exist_ok=True)
    
    print(f"Saving generated image at {image_path}")
    image.save(str(image_path))
    print("Image saved successfully")


def generate_diffusion_pipe(device, dtype, model_path=None, lora=None):
    """generate the diffusion pipeline"""

    # will download the model if it isn't found in the .cache/huggingface/hub folder directory.

    repo_id = "OnomaAIResearch/Illustrious-XL-v1.1"
    filename = "Illustrious-XL-v1.1.safetensors"

    if model_path:
        model_path = Path(f"models/{model_path}")
    else:
        try:
            # try downloading the model from the cache
            print("Checking cache for model")
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                local_files_only=True,  # Don't download, just check cache
            )
            print(f"Successfully found model: {repo_id}/{filename}")
        except Exception:
            # downloading model if not found in cache
            print("Model not found in cache. Downloading (~7GB for first download)")
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model"
            )
            print(f"Successfully downloaded model: {repo_id}/{filename}. Downloaded to -> {model_path}")

    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=dtype,
        use_safetensors=True,
        add_watermarker=False,
    )

    # move to cuda if available
    pipe = pipe.to(device)

    # optional memory optimisations
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    # optional if a LoRA adapter is specified
    if lora:
        path_to_lora = Path("lora_adapters/" + lora)
        print(f"Adding LoRA weights: {path_to_lora}")
        pipe.load_lora_weights(path_to_lora)

    return pipe


def show_all_adapters():
    "show a list of all available LoRA adapters"

    adapters_dir = Path("lora_adapters")

    # find the contents of this directory if it exists
    if os.path.isdir(adapters_dir):
        list_of_adapters = os.listdir(adapters_dir)
    else:
        list_of_adapters = None

    return list_of_adapters


def parse_arguments():
    "parsing command line arguments"

    parser = argparse.ArgumentParser(
        description="Generate images using a locally hosted diffusion model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "prompt", help="prompt to apply to image generation model", type=str
    )
    parser.add_argument(
        "-n",
        "--negative_prompt",
        help="negative prompt to apply to image generation model",
        type=str,
    )
    parser.add_argument(
        "-l", "--lora", help="apply a LoRA adapter to the output", type=str
    )
    parser.add_argument(
        "--model_path", help="provide a custom local model path within the models directory", type=str
    )
    parser.add_argument(
        "--num_inference_steps", help="number of inference steps", type=int, default=25
    )
    parser.add_argument(
        "--guidance_scale",
        help="how closely the image generation adheres to the prompt",
        type=float,
        default=0.7,
    )
    parser.add_argument("--width", help="image width", default=1024)
    parser.add_argument("--height", help="image height", default=1024)
    parser.add_argument("--seed", help="random seed", default=None)
    parser.add_argument("--image_name", help="saved image name", default="image")
    parser.add_argument(
        "--list",
        help="list available LoRA adapters",
        action="store_true",
        default=False,
    )
    parser.add_argument("--save_gif", help="save progress gif", action='store_true', default=False)
    parser.add_argument("--gif_name", help="saved gif name", type=str)
    return parser.parse_args()


def main():
    "main script logic"

    args = parse_arguments()
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    lora = args.lora
    model_path = args.model_path
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    width = args.width
    height = args.height
    seed = args.seed
    image_name = args.image_name
    save_gif = args.save_gif
    gif_name = args.gif_name

    show_list = args.list

    if show_list:
        list_of_adapters = show_all_adapters()
        if list_of_adapters:
            print("List of all available LoRA adapters: ", ", ".join(list_of_adapters))
        else:
            print("No directory 'lora_adapters' found")

    # if the save_gif argument is passed without a name for the gif
    gif_name = args.gif_name if args.gif_name else args.image_name

    device = check_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = generate_diffusion_pipe(device, dtype, model_path, lora)

    image, intermediate_images = generate_image(
        pipe,
        prompt,
        negative_prompt,
        num_inference_steps,
        guidance_scale,
        width,
        height,
        device,
        seed,
        save_gif,
        gif_name
    )

    save_image(image, image_name)

    if save_gif and intermediate_images:
        gif_path = Path('images') / f"{gif_name}.gif"
        callback_utils.save_progress_gif(pipe, intermediate_images, gif_path)


if __name__ == "__main__":
    main()
