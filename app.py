import torch
from PIL import Image
import numpy as np
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample)

import streamlit as st
import random

import logging as logger

from typing import Tuple

import nltk 
nltk.download('popular')

def convert_to_images(obj):
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
        img.append(Image.fromarray(out_array))
    return img


def save_as_images(obj, file_name='output'):
    """ Convert and save an output tensor from BigGAN in a list of saved images.
        Params:
            obj: tensor or numpy array of shape (batch_size, channels, height, width)
            file_name: path and beggingin of filename to save.
                Images will be saved as `file_name_{image_number}.png`
    """
    img = convert_to_images(obj)

    for i, out in enumerate(img):
        current_file_name = file_name + '_%d.png' % i
        logger.info("Saving image to {}".format(current_file_name))
        out.save(current_file_name, 'png')

@st.cache
def load_model(pretrained_name: str):
    model = BigGAN.from_pretrained(pretrained_name)
    return model

class StreamlintGanModel:

    batch_size: int = 1
    truncation: float = 0.4

    def __init__(self, pretrained_name: str="biggan-deep-256", device: str="cuda"):
        self.pretrained_name = pretrained_name
        self.device = device
        self.model = load_model(self.pretrained_name).to(self.device)

    @staticmethod
    def load_model(pretrained_name: str):
        model = BigGAN.from_pretrained(pretrained_name)
        return model

    def preprocessing(self, name: str, seed: int=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seed is None:
            seed = random.randint(1, 10000)

        class_vector = torch.from_numpy(one_hot_from_names([name], batch_size=self.batch_size))
        noise_vector = torch.from_numpy(truncated_noise_sample(truncation=self.truncation, batch_size=self.batch_size, seed=seed))
        return (class_vector, noise_vector)

    @torch.no_grad()
    def predict(self, name: str, seed: int=None) -> torch.Tensor:
        class_vector, noise_vector = self.preprocessing(name, seed)
        output = self.model(noise_vector, class_vector, self.truncation).to("cpu")
        return output


model = StreamlintGanModel(pretrained_name="biggan-deep-256", device="cpu")

st.markdown("### Homework BigGan Buts")

name = st.text_input("Which object do you want to generate?")
seed = st.number_input("Choose seed")

if name is not None and st.button('Generate'):
    if seed is not None and (seed <= 0 or seed > 10000):
        seed = None
    
    output = model.predict(name="cup", seed=None)
    save_as_images(output, file_name="output")
    st.image("output_0.png")
