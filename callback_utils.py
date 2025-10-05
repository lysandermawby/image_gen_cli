#!/usr/bin/python
"""utilities for generating and saving a gif of generation"""

# imports
import torch
from PIL import Image
import numpy as np
from print_utils import print_error, print_success, print_debug
import traceback
from tqdm import tqdm

def create_progress_callback():
    """Create a callback that captures intermediate steps of image generation"""
    intermediate_images = []

    def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
        """called at every diffusion step, saving latents"""
        # Get latents from callback_kwargs
        latents = callback_kwargs.get("latents")
        if latents is not None:
            intermediate_images.append({
                'step': step_index,
                'latents': latents.clone()
            })

        return callback_kwargs
    
    return callback_on_step_end, intermediate_images


def save_progress_gif(pipe, intermediate_images, output_path):
    """Convert intermediate latents to images and save as GIF"""
    
    print(f"Generating progress GIF with {len(intermediate_images)} frames...")
    
    latents_list = []
    for item in intermediate_images:
        try:
            with torch.no_grad():
                latents = item['latents']
                # scale latents
                latents = 1 / 0.18215 * latents
                latents_list.append(latents)
        except Exception as e:
            print_error(f"Error processing latents: {e}")
            traceback.print_exc()
            raise ValueError
    
    # batch size set to avoid memory issues on a CPU
    batch_size = 1 
    decoding_pbar = tqdm(range(0, len(latents_list), batch_size), desc = "Decoding latents", unit='batch')
    decoded_images_list = []
    for i in decoding_pbar:
        batch = latents_list[i:i + batch_size]
        batch_latents = torch.cat(batch, dim=0)

        with torch.no_grad():
            decoded_batch = pipe.vae.decode(batch_latents).sample
            decoded_images_list.append(decoded_batch)

    decoded_images = torch.cat(decoded_images_list, dim=0)
    decoded_images = (decoded_images / 2 + 0.5).clamp(0, 1)
    decoded_images = decoded_images.cpu().permute(0, 2, 3, 1).numpy()
    decoded_images = (decoded_images * 255).astype(np.uint8)

    frames = []
    for image in decoded_images:
        frames.append(Image.fromarray(image))
    
    # Save as GIF
    print_debug(f"About to save GIF frames. Found {len(frames)} to save")
    if frames:
        print_debug("Saving GIF frames in save_progress_gif")
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=200,  # ms per frame
            loop=0
        )
        print_success(f"Progress GIF saved to {output_path}")
