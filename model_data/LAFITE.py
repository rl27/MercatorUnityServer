#!/usr/bin/env python

from model_data.hyperbolic_generative_model import HyperbolicGenerativeModel
import torch
import numpy as np
from PIL import Image
import clip
import sys
sys.path.insert(1, 'Lafite')
from Lafite import dnnlib, legacy

import gdown

use_cpu = True

class Generator:
    def __init__(self, device, path):
        self.name = 'generator'
        self.model = self.load_model(device, path)
        self.device = device
        self.force_32 = use_cpu # True if using CPU, false if using GPU
        
    def load_model(self, device, path):
        with dnnlib.util.open_url(path) as f:
            network= legacy.load_network_pkl(f)
            self.G_ema = network['G_ema'].to(device)
            self.D = network['D'].to(device)
            # self.G = network['G'].to(device)
            return self.G_ema

    def generate(self, z, c, fts, noise_mode='const', return_styles=True):
        return self.model(z, c, fts=fts, noise_mode=noise_mode, return_styles=return_styles, force_fp32=self.force_32)
    
    def generate_from_style(self, style, noise_mode='const'):
        ws = torch.randn(1, self.model.num_ws, 512)
        return self.model.synthesis(ws, fts=None, styles=style, noise_mode=noise_mode, force_fp32=self.force_32)
    
    def tensor_to_img(self, tensor):
        img = torch.clamp((tensor + 1.) * 127.5, 0., 255.)
        img_list = img.permute(0, 2, 3, 1)
        img_list = [img for img in img_list]
        return Image.fromarray(torch.cat(img_list, dim=-2).detach().cpu().numpy().astype(np.uint8))

class PoincareLAFITE(HyperbolicGenerativeModel):

    latent_dim = 64

    def __init__(self):
        #gdown.download('https://drive.google.com/uc?id=1eNkuZyleGJ3A3WXTCIGYXaPwJ6NH9LRA', 'LAFITE_G.pkl')
        #gdown.download('https://drive.google.com/u/0/uc?id=1WQnlCM4pQZrw3u9ZeqjeUNqHuYfiDEU3', 'LAFITE_NN.pkl')
        #gdown.download('https://drive.google.com/u/0/uc?id=17ER7Yl02Y6yCPbyWxK_tGrJ8RKkcieKq', 'LAFITE_CC.pkl')
        
        with torch.no_grad():
            self.device = 'cpu' if use_cpu else 'cuda:0'
            path = './LAFITE_NN.pkl'  # pre-trained model
            self.generator = Generator(device=self.device, path=path)
            self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
            self.clip_model = self.clip_model.eval()

    def generate_image_from_sentence(self, sentence):
        with torch.no_grad():
            tokenized_text = clip.tokenize([sentence]).to(self.device)
            txt_fts = self.clip_model.encode_text(tokenized_text)
            txt_fts = txt_fts / txt_fts.norm(dim=-1, keepdim=True)
            
            # These were previously randn in the LAFITE example
            z = torch.zeros((1, 512)).to(self.device)
            c = torch.zeros((1, 1)).to(self.device) # label is actually not used
            img, _ = self.generator.generate(z=z, c=c, fts=txt_fts)
            im = self.generator.tensor_to_img(img)
        return im, txt_fts[0].cpu().tolist()

    def generate_image_from_latent_vector(self, v) -> Image:
        # [1, 512]
        v = torch.tensor(v, dtype=torch.float).to(self.device)

        with torch.no_grad():
            # These were previously randn in the LAFITE example
            z = torch.zeros((1, 512)).to(self.device)
            c = torch.zeros((1, 1)).to(self.device) # label is actually not used
            img, _ = self.generator.generate(z=z, c=c, fts=v)
            im = self.generator.tensor_to_img(img)

        return im

    def generate_multiple(self, v):
        # [n, 512]
        v = torch.tensor(v, dtype=torch.float).to(self.device)
        n = v.shape[0]

        with torch.no_grad():
            z = torch.zeros((n, 512)).to(self.device)
            c = torch.zeros((n, 1)).to(self.device) # label is actually not used
            imgs, _ = self.generator.generate(z=z, c=c, fts=v)

            images = []
            for i in range(n):
                images.append(self.generator.tensor_to_img(imgs[i].unsqueeze(dim=0)))

        return images