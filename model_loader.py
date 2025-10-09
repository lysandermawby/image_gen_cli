#!/usr/bin/python
"""Model loading utilities for SDXL pipelines"""

from pathlib import Path
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch


MODELS = {
    # Anime/Illustration Models
    "illustrious": {
        "repo_id": "OnomaAIResearch/Illustrious-XL-v1.1",
        "filename": "Illustrious-XL-v1.1.safetensors",
        "description": "High-quality anime model with extensive training data"
    },
    "animagine-4": {
        "repo_id": "cagliostrolab/animagine-xl-4.0",
        "filename": None,
        "description": "Latest Animagine version with improved prompt understanding"
    },
    "animagine-3": {
        "repo_id": "Linaqruf/animagine-xl",
        "filename": None,
        "description": "Popular anime model with Danbooru-style tagging"
    },

    # Official Stability AI Models
    "sdxl-base": {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "filename": "sd_xl_base_1.0.safetensors",
        "description": "Official SDXL base model for general purpose generation"
    },
    "sdxl-refiner": {
        "repo_id": "stabilityai/stable-diffusion-xl-refiner-1.0",
        "filename": "sd_xl_refiner_1.0.safetensors",
        "description": "Refiner model for adding final details to base model outputs"
    },
    "sdxl-turbo": {
        "repo_id": "stabilityai/sdxl-turbo",
        "filename": None,
        "description": "Fast 1-4 step generation model optimized for speed"
    },
    "japanese-sdxl": {
        "repo_id": "stabilityai/japanese-stable-diffusion-xl",
        "filename": None,
        "description": "SDXL fine-tuned for Japanese language and culture"
    },

    # Photorealistic Models
    "juggernaut-xl": {
        "repo_id": "RunDiffusion/Juggernaut-XL-v9",
        "filename": None,
        "description": "Popular photorealistic model with 520K+ downloads"
    },
    "dreamshaper-xl": {
        "repo_id": "Lykon/dreamshaper-xl-1-0",
        "filename": None,
        "description": "General purpose model for photos, art, anime, and manga"
    },
    "realvis-xl": {
        "repo_id": "SG161222/RealVisXL_V4.0",
        "filename": None,
        "description": "Hyper-realistic model specializing in human figures"
    },

    # Speed-Optimized Models (Lightning)
    "sdxl-lightning-1step": {
        "repo_id": "ByteDance/SDXL-Lightning",
        "filename": "sdxl_lightning_1step_unet_x0.safetensors",
        "description": "1-step distilled SDXL (experimental, lower quality)",
        "is_unet_only": True
    },
    "sdxl-lightning-2step": {
        "repo_id": "ByteDance/SDXL-Lightning",
        "filename": "sdxl_lightning_2step_unet.safetensors",
        "description": "2-step distilled SDXL with good quality",
        "is_unet_only": True
    },
    "sdxl-lightning-4step": {
        "repo_id": "ByteDance/SDXL-Lightning",
        "filename": "sdxl_lightning_4step_unet.safetensors",
        "description": "4-step distilled SDXL with excellent quality",
        "is_unet_only": True
    },
    "sdxl-lightning-8step": {
        "repo_id": "ByteDance/SDXL-Lightning",
        "filename": "sdxl_lightning_8step_unet.safetensors",
        "description": "8-step distilled SDXL with best quality",
        "is_unet_only": True
    },

    # Speed-Optimized Models (Hyper-SD)
    "hyper-sdxl-1step": {
        "repo_id": "ByteDance/Hyper-SD",
        "filename": "Hyper-SDXL-1step-Unet.safetensors",
        "description": "Hyper-SD 1-step UNet for ultra-fast generation",
        "is_unet_only": True
    },

    # Additional Anime Models
    "pony-diffusion-xl": {
        "repo_id": "AstraliteHeart/pony-diffusion-v6-xl",
        "filename": None,
        "description": "Anime model with unique prompting style (use score tags)"
    },
    "noobai-xl": {
        "repo_id": "Corcelio/noobai-XL-Vpred-1.1",
        "filename": None,
        "description": "Recent anime model based on Illustrious-XL"
    },
}


def print_available_models():
    """Print all available models organized by category"""
    print("\nAnime/Illustration Models:")
    for key in ["illustrious", "animagine-4", "animagine-3", "pony-diffusion-xl", "noobai-xl"]:
        if key in MODELS:
            print(f"  - {key}: {MODELS[key]['description']}")

    print("\nOfficial Stability AI Models:")
    for key in ["sdxl-base", "sdxl-refiner", "sdxl-turbo", "japanese-sdxl"]:
        if key in MODELS:
            print(f"  - {key}: {MODELS[key]['description']}")

    print("\nPhotorealistic Models:")
    for key in ["juggernaut-xl", "dreamshaper-xl", "realvis-xl"]:
        if key in MODELS:
            print(f"  - {key}: {MODELS[key]['description']}")

    print("\nSpeed-Optimized Models:")
    for key in [k for k in MODELS.keys() if "lightning" in k or "hyper" in k]:
        print(f"  - {key}: {MODELS[key]['description']}")


def load_custom_model(model_path, dtype):
    """Load model from custom local path"""
    model_path = Path(f"models/{model_path}")
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=dtype,
        use_safetensors=True,
        add_watermarker=False,
    )
    return pipe


def load_unet_only_model(repo_id, filename, dtype, device):
    """Load UNet-only models (Lightning/Hyper-SD) with base SDXL components"""
    print("This is a UNet-only model. Loading with SDXL base components...")

    try:
        print("Checking cache for UNet model")
        unet_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            local_files_only=True,
        )
        print(f"Successfully found model: {repo_id}/{filename}")
    except Exception:
        print("Model not found in cache. Downloading...")
        unet_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model"
        )
        print(f"Successfully downloaded model: {repo_id}/{filename}")

    # Load base SDXL with custom UNet
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, dtype)
    unet.load_state_dict(load_file(unet_path, device=device))

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base,
        unet=unet,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
        add_watermarker=False,
    )

    return pipe


def load_single_file_model(repo_id, filename, dtype):
    """Load single-file models (.safetensors)"""
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
    return pipe


def load_repository_model(repo_id, dtype):
    """Load models from HuggingFace repository"""
    print(f"Loading model from repository: {repo_id}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        repo_id,
        torch_dtype=dtype,
        add_watermarker=False,
        use_safetensors=True,
    )
    return pipe


def load_model(device, dtype, model_path=None, model_name=None):
    """
    Main function to load appropriate SDXL pipeline based on parameters

    Args:
        device: Device to load model on ('cuda' or 'cpu')
        dtype: Data type for model (torch.float16 or torch.float32)
        model_path: Optional custom local model path
        model_name: Name of predefined model from MODELS dict

    Returns:
        Loaded StableDiffusionXLPipeline or None if invalid model
    """
    # Handle model name validation
    if model_name is None:
        model_name = "illustrious"
        print(f"No model name provided. Using {model_name} by default")
    elif model_name not in MODELS.keys():
        print(f"Model '{model_name}' is not available. Please choose from the following:")
        print_available_models()
        return None
    else:
        print(f"Model {model_name} successfully located")
        print(f"Description: {MODELS[model_name]['description']}")

    # Extract model configuration
    repo_id = MODELS[model_name]["repo_id"]
    filename = MODELS[model_name].get("filename")
    is_unet_only = MODELS[model_name].get("is_unet_only", False)

    # Load model based on type
    if model_path:
        pipe = load_custom_model(model_path, dtype)
    elif is_unet_only:
        pipe = load_unet_only_model(repo_id, filename, dtype, device)
        # Configure scheduler for Lightning models
        if "lightning" in model_name:
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config,
                timestep_spacing="trailing"
            )
            steps = model_name.split('-')[2]
            print(f"Note: Use {steps} inference steps with guidance_scale=0")
    elif filename:
        pipe = load_single_file_model(repo_id, filename, dtype)
    else:
        pipe = load_repository_model(repo_id, dtype)
        # Special configuration for SDXL-Turbo
        if model_name == "sdxl-turbo":
            print("Note: SDXL-Turbo works best with 1-4 steps and guidance_scale=0.0")

    return pipe
