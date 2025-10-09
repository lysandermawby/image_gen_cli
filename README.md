# imggen - Local AI Image Generation CLI

A CLI for generating images locally using Stable Diffusion XL models with optional LoRA adapters. All processing is conducted locally.

## Features

- **Multiple Model Support** - Supports Stability AI Models, Juggernaut photorealistic models, and Speed optimised models
- **Easily Accessible CLI** - Can be run from the terminal at any time, generating and saving images locally
- **GIFs of Generations** - Allows GIFs of the generations to be saved 

All of this using exclusively locally downloaded models and local compute.

## Setup

Run the `setup.sh` script to move to code to a `~/.image-gen/` directory and to create a `imggen` command in your shell.

```bash
chmod +x setup.sh
./setup.sh
```

Any models downloaded will be stored in your huggingface cache. 
LoRA adapters which you want to apply should be downloaded to the `~/image-gen/lora_adapters/` directory.

Add this to your system path with the following command to ensure that this `imggen` CLI persists across shell sessions.

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Generating An Image

To generate an image, run the `imggen` CLI with your chosen prompt. Use the following options to generate an image and display it when completed

```bash
Usage: imggen [OPTIONS] [PROMPT]

  Generate images using a locally hosted diffusion model

Options:
  -n, --negative-prompt TEXT     negative prompt to apply to image generation
                                 model
  -l, --lora TEXT                apply a LoRA adapter to the output
  --model-path TEXT              provide a custom local model path within the
                                 models directory
  --model-name TEXT              provide a model name to be used from cache
  --num-inference-steps INTEGER  number of inference steps
  --guidance-scale FLOAT         how closely the image generation adheres to
                                 the prompt
  --width INTEGER                image width
  --height INTEGER               image height
  --seed INTEGER                 random seed
  --image-name TEXT              saved image name
  --save-gif                     save progress gif
  --gif-name TEXT                saved gif name
  --no-display                   suppress image display
  --list                         list available LoRA adapters
  -v, --version                  show script version
  -h, --help                     Show this message and exit.
```

For example, run the following command to generate a set of rolling hills with a Stability AI model.

```bash
# WARNING: This will download the stable-diffusion-xl-base-1.0 model to your Huggingface cache if it isn't already present
imggen "Rolling hills in spring" --negative-prompt "bad hands, watermark, unclear, block colour, text, pixelated" --model-name sdxl-base --num-inference-steps 20 --guidance-scale 12 --image-name "rolling_hills" --save-gif
```

This will save images (and GIFs of generation if the --save-gif argument is passed) to the `~/.image-gen/images` directory by default.
