#!/usr/bin/env python

import torch
from model_data.hyperbolic_generative_model import HyperbolicGenerativeModel
from PIL import Image
import numpy as np

class PoincareGANzoo(HyperbolicGenerativeModel):
    # 120 for DCGAN, 512 for PGAN
    latent_dim = 512

    def __init__(self):
        # PGAN models: ['celebAHQ-256', 'celebAHQ-512', 'DTD', 'celeba']

        # Uncomment the line below to use cpu instead of gpu
        # torch.cuda.is_available = lambda : False

        use_gpu = True if torch.cuda.is_available() else False
        _device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(_device)
        self.model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                                    'PGAN', model_name='celebAHQ-512',
                                     pretrained=True, useGPU=use_gpu, map_location=torch.device(_device))


    def generate_image_from_latent_vector(self, v) -> Image:
        
        # Normalization; pytorch_GAN_zoo already does normalization
        # https://github.com/facebookresearch/pytorch_GAN_zoo/blob/main/models/networks/custom_layers.py#L9
        # v = np.array(v)
        # v = v / np.sqrt(np.sum(v**2))
        # v = v / np.sqrt(np.sum(v**2, axis=1, keepdims=True))

        # To generate random inputs, see get_random_coords in ImageSampler. Alternatively:
        # https://github.com/facebookresearch/pytorch_GAN_zoo/blob/main/models/base_GAN.py#L328
        # v = torch.randn(1, latent_dim).to('cuda')

        v = torch.tensor(v, dtype=torch.float).to(self.device).unsqueeze(dim=0)

        with torch.no_grad():
            outputs = self.model.test(v, toCPU=True)

        images = self.convert_to_images(outputs)
        return images[0]

    def generate_multiple(self, v) -> Image:

        v = torch.tensor(v, dtype=torch.float).to(self.device)

        with torch.no_grad():
            generated_images = self.model.test(v, toCPU=True)

        images = self.convert_to_images(generated_images)
        return images


    def convert_to_images(self, obj):
        """ Convert an output tensor from BigGAN in a list of images.
            Params:
                obj: tensor or numpy array of shape (batch_size, channels, height, width)
            Output:
                list of Pillow Images of size (height, width)
        """
        try:
            import PIL
        except ImportError:
            raise ImportError("Please install Pillow to use images: pip install Pillow")

        if not isinstance(obj, np.ndarray):
            obj = obj.detach().numpy()

        obj = obj.transpose((0, 2, 3, 1))
        obj = np.clip(((obj + 1) / 2.0) * 256, 0, 255)

        img = []
        for i, out in enumerate(obj):
            out_array = np.asarray(np.uint8(out), dtype=np.uint8)
            img.append(PIL.Image.fromarray(out_array))
        return img
