from pathlib import Path

import sd_config
import os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from imwatermark import WatermarkEncoder
from concurrent.futures import Future
from typing import Union, Any

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

torch.set_grad_enabled(False)

config = OmegaConf.load(f"{sd_config.CONFIG}")


def load_model_from_config(config=config, ckpt=f"{sd_config.CKPT}", device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()

    return model


device = torch.device("cuda") if sd_config.DEVICE == "cuda" else torch.device("cpu")
model = load_model_from_config()


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


# Generate images
def generate(
    outputPath: Path,
    prompt: str,
    height: int,
    width: int,
    steps: int,
    seed: int,
    channels: int = sd_config.CHANNELS
) -> None:

    seed_everything(seed)

    if sd_config.PLMS:
        sampler = PLMSSampler(model, device=device)
    elif sd_config.DPM:
        sampler = DPMSolverSampler(model, device=device)
    else:
        sampler = DDIMSampler(model, device=device)

    os.makedirs(sd_config.OUTDIR, exist_ok=True)
    outpath = sd_config.OUTDIR

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = sd_config.N_SAMPLES
    n_rows = sd_config.N_ROWS if sd_config.N_ROWS > 0 else batch_size
    if not sd_config.FROM_FILE:
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {sd_config.FROM_FILE}")
        with open(sd_config.FROM_FILE, "r") as f:
            data = f.read().splitlines()
            data = [p for p in data for i in range(sd_config.REPEAT)]
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    start_code = None
    if sd_config.FIXED_CODE:
        start_code = torch.randn([sd_config.N_SAMPLES, sd_config.CHANNELS, sd_config.HEIGHT // sd_config.DOWNSAPLE_FACTOR, sd_config.WIDTH // sd_config.DOWNSAPLE_FACTOR], device=device)

    if sd_config.TORCHSCRIPT or sd_config.IPEX:
        transformer = model.cond_stage_model.model
        unet = model.model.diffusion_model
        decoder = model.first_stage_model.decoder
        additional_context = torch.cpu.amp.autocast() if sd_config.BF16 else nullcontext()
        shape = [sd_config.CHANNELS, sd_config.HEIGHT // sd_config.DOWNSAPLE_FACTOR, sd_config.WIDTH // sd_config.DOWNSAPLE_FACTOR]

        if sd_config.BF16 and not sd_config.TORCHSCRIPT and not sd_config.IPEX:
            raise ValueError('Bfloat16 is supported only for torchscript+ipex')
        if sd_config.BF16 and unet.dtype != torch.bfloat16:
            raise ValueError("Use configs/stable-diffusion/intel/ configs with bf16 enabled if " +
                             "you'd like to use bfloat16 with CPU.")
        if unet.dtype == torch.float16 and device == torch.device("cpu"):
            raise ValueError("Use configs/stable-diffusion/intel/ configs for your model if you'd like to run it on CPU.")

        if sd_config.IPEX:
            import intel_extension_for_pytorch as ipex
            bf16_dtype = torch.bfloat16 if sd_config.BF16 else None
            transformer = transformer.to(memory_format=torch.channels_last)
            transformer = ipex.optimize(transformer, level="O1", inplace=True)

            unet = unet.to(memory_format=torch.channels_last)
            unet = ipex.optimize(unet, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

            decoder = decoder.to(memory_format=torch.channels_last)
            decoder = ipex.optimize(decoder, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

        if sd_config.TORCHSCRIPT:
            with torch.no_grad(), additional_context:
                # get UNET scripted
                if unet.use_checkpoint:
                    raise ValueError("Gradient checkpoint won't work with tracing. " +
                    "Use configs/stable-diffusion/intel/ configs for your model or disable checkpoint in your config.")

                img_in = torch.ones(2, 4, 96, 96, dtype=torch.float32)
                t_in = torch.ones(2, dtype=torch.int64)
                context = torch.ones(2, 77, 1024, dtype=torch.float32)
                scripted_unet = torch.jit.trace(unet, (img_in, t_in, context))
                scripted_unet = torch.jit.optimize_for_inference(scripted_unet)
                print(type(scripted_unet))
                model.model.scripted_diffusion_model = scripted_unet

                # get Decoder for first stage model scripted
                samples_ddim = torch.ones(1, 4, 96, 96, dtype=torch.float32)
                scripted_decoder = torch.jit.trace(decoder, (samples_ddim))
                scripted_decoder = torch.jit.optimize_for_inference(scripted_decoder)
                print(type(scripted_decoder))
                model.first_stage_model.decoder = scripted_decoder

        prompts = data[0]
        print("Running a forward pass to initialize optimizations")
        uc = None
        if sd_config.SCALE != 1.0:
            uc = model.get_learned_conditioning(batch_size * [""])
        if isinstance(prompts, tuple):
            prompts = list(prompts)

        with torch.no_grad(), additional_context:
            for _ in range(3):
                c = model.get_learned_conditioning(prompts)
            samples_ddim, _ = sampler.sample(S=5,
                                             conditioning=c,
                                             batch_size=batch_size,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=sd_config.SCALE,
                                             unconditional_conditioning=uc,
                                             eta=sd_config.DDIM_ETA,
                                             x_T=start_code)
            print("Running a forward pass for decoder")
            for _ in range(3):
                x_samples_ddim = model.decode_first_stage(samples_ddim)

    precision_scope = autocast if sd_config.PRECISION=="autocast" or sd_config.BF16 else nullcontext
    with torch.no_grad(), \
        precision_scope(sd_config.DEVICE), \
        model.ema_scope():
            all_samples = list()
            for n in trange(sd_config.N_ITER, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if sd_config.SCALE != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [channels, height // sd_config.DOWNSAPLE_FACTOR, width // sd_config.DOWNSAPLE_FACTOR]
                    samples, _ = sampler.sample(S=steps,
                                                     conditioning=c,
                                                     batch_size=sd_config.N_SAMPLES,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=sd_config.SCALE,
                                                     unconditional_conditioning=uc,
                                                     eta=sd_config.DDIM_ETA,
                                                     x_T=start_code)

                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)


                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img = put_watermark(img, wm_encoder)
                        img.save(outputPath)

                    all_samples.append(x_samples)
