import os
import platform
from PIL import Image
from IPython.display import display
import torch

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

from googletrans import Translator

if platform.system() == "Windows":
    base_path = str(os.path.abspath(__file__)).split('\\')
else:
    base_path = str(os.path.abspath(__file__)).split('/')
base_path = '/'.join(base_path[:-1])

model_path = base_path + '/models/painterbot/model'
diffusion_path = base_path + '/models/painterbot/diffusion'
save_image_path = base_path + '/models/painterbot/images/temp.jpeg'

has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')


def save_images(batch: torch.Tensor):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0, 255).to(torch.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    output_image = Image.fromarray(reshaped.numpy())
    output_image.save(save_image_path)
    return Image.fromarray(reshaped.numpy())


def painterbot(prompt):

    # Google translator 라이브러리를 통한 번역
    translator = Translator()
    prompt = translator.translate(prompt, dest='en')
    prompt = str(prompt.text)
    print(prompt)

    batch_size = 1
    guidance_scale = 3.0

    model = torch.load(model_path)
    model.eval()
    tokenizer = model.tokenizer
    diffusion = torch.load(diffusion_path)
    options = model_and_diffusion_defaults_upsampler()
    options['use_fp16'] = has_cuda
    # use 27 diffusion steps for very fast sampling
    options['timestep_respacing'] = 'fast27'

    # Create a classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    # Create the text tokens to feed to the model.
    tokens = tokenizer.encode(prompt)
    tokens, mask = tokenizer.padded_tokens_and_mask(
        tokens, options['text_ctx']
    )

    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = tokenizer.padded_tokens_and_mask(
        [], options['text_ctx']
    )

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=torch.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask=torch.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=torch.bool,
            device=device,
        ),
    )

    # Output image from the model.
    model.del_cache()
    output = diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, options["image_size"], options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    model.del_cache()

    return save_images(output)
