#!/usr/bin/python
"""Generate images using the Illustrious-XL model"""

# package imports
import torch
from diffusers import StableDiffusionXLPipeline
import click
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import subprocess
import re

# local imports
import callback_utils
from print_utils import print_error

# script variables
VERSION = "0.1.0"


def show_version():
    """shwo the version"""
    print(VERSION)
    return


def display_object(image_path):
    """display the generated image / gif with the default viewer"""
    subprocess.run(['open', str(image_path)])
    return


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
    return image_path


def generate_diffusion_pipe(device, dtype, model_path=None, model_name=None, lora=None):
    """generate the diffusion pipeline"""

    # will download the model if it isn't found in the .cache/huggingface/hub folder directory.

    MODELS = {
        "illustrious": {
            "repo_id": "OnomaAIResearch/Illustrious-XL-v1.1",
            "filename": "Illustrious-XL-v1.1.safetensors"
        },
        "animagine-4": {
            "repo_id": "cagliostrolab/animagine-xl-4.0",
            "filename": None  # uses from_pretrained directly
        },
        "animagine-3": {
            "repo_id": "Linaqruf/animagine-xl",
            "filename": None
        }
    }

    if model_name is None:
        model_name = "illustrious"
        print(f"No model name provided. Using {model_name} by default")
    elif model_name not in MODELS.keys():
        print(f"Model provided is not available. Please choose one of the following available models: {', '.join(MODELS.keys())}")
    else:
        print(f"Model {model_name} successfully located and deployed")

    repo_id = MODELS[model_name]["repo_id"]
    filename = MODELS[model_name]["filename"]

    # repo_id = "OnomaAIResearch/Illustrious-XL-v1.1"
    # filename = "Illustrious-XL-v1.1.safetensors"

    if model_path:
        # Custom local model
        model_path = Path(f"models/{model_path}")
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=dtype,
            use_safetensors=True,
            add_watermarker=False,
        )
    elif filename:
        # Single-file model (like illustrious)
        try:
            print("Checking cache for model")
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                local_files_only=True,
            )
            print(f"Successfully found model: {repo_id}/{filename}")
        except Exception:
            print("Model not found in cache. Downloading (~7GB for first download)")
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model"
            )
            print(f"Successfully downloaded model: {repo_id}/{filename}")

        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=dtype,
            use_safetensors=True,
            add_watermarker=False,
        )
    else:
        # Repository model (like animagine)
        print(f"Loading model from repository: {repo_id}")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            repo_id,
            torch_dtype=dtype,
            add_watermarker=False,
            use_safetensors=True,
        )

    # move to device and ensure consistent dtype
    pipe = pipe.to(device)

    # force all model components to have a consistent data type. Required for CPU processing
    if device == "cpu":
        pipe = pipe.to(torch.float32)

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
        # Filter for .safetensors files
        list_of_adapters = [x for x in list_of_adapters if x.endswith('.safetensors')]
    else:
        list_of_adapters = None

    return list_of_adapters


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument("prompt", required=False)
@click.option("-n", "--negative-prompt", type=str, help="negative prompt to apply to image generation model")
@click.option("-l", "--lora", type=str, help="apply a LoRA adapter to the output")
@click.option("--model-path", type=str, help="provide a custom local model path within the models directory")
@click.option("--model-name", type=str, help="provide a model name to be used from cache")
@click.option("--num-inference-steps", type=int, default=25, help="number of inference steps")
@click.option("--guidance-scale", type=float, default=12, help="how closely the image generation adheres to the prompt")
@click.option("--width", type=int, default=1024, help="image width")
@click.option("--height", type=int, default=1024, help="image height")
@click.option("--seed", type=int, default=None, help="random seed")
@click.option("--image-name", type=str, default="image", help="saved image name")
@click.option("--save-gif", is_flag=True, help="save progress gif")
@click.option("--gif-name", type=str, default=None, help="saved gif name")
@click.option("--no-display", is_flag=True, help="suppress image display")
@click.option("--list", "show_list", is_flag=True, help="list available LoRA adapters")
@click.option("-v", "--version", "show_version", is_flag=True, help="show script version")
def main(prompt, negative_prompt, lora, model_path, model_name, num_inference_steps,
         guidance_scale, width, height, seed, image_name, save_gif, gif_name,
         no_display, show_list, show_version):
    """Generate images using a locally hosted diffusion model"""

    if show_version:
        print(VERSION)
        return

    if show_list:
        list_of_adapters = show_all_adapters()
        if list_of_adapters:
            print("List of all available LoRA adapters: ", ", ".join(list_of_adapters))
        else:
            print("No directory 'lora_adapters' found")
        return

    if not prompt:
        print_error("Error: Prompt is required for image generation")
        return 1

    # if the save_gif argument is passed without a name for the gif
    gif_name = gif_name if gif_name else image_name

    device = check_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = generate_diffusion_pipe(device, dtype, model_path, model_name, lora)

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

    # save and display image
    image_path = save_image(image, image_name)
    if not no_display:
        print(f"Opening generated image from: {image_path}")
        display_object(image_path)

    if save_gif and intermediate_images:
        gif_path = Path('images') / f"{gif_name}.gif"
        callback_utils.save_progress_gif(pipe, intermediate_images, gif_path)
        if not no_display:
            print(f"Opening generated gif from: {gif_path}")
            display_object(gif_path)


if __name__ == "__main__":
    main()
