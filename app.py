import torch
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)

import streamlit as st
import random

from typing import Tuple

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
    
    output = model.predict(name=name, seed=seed)
    save_as_images(output, file_name="output")
    st.image("output")
