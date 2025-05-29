from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torchdiffeq import odeint
from tqdm import tqdm
import numpy as np
from lutils.configuration import Configuration
from lutils.dict_wrapper import DictWrapper
from model.vector_field_regressor import build_vector_field_regressor
from model.vqgan.taming.autoencoder import vq_f8_ddconfig, vq_f8_small_ddconfig, vq_f16_ddconfig, VQModelInterface
from model.vqgan.vqvae import build_vqvae

def inner_prod(x, y):
    x = x.reshape((x.shape[0], -1))
    y = y.reshape((y.shape[0], -1))
    return torch.sum(x * y, axis=-1, keepdims=True)

class Model(nn.Module):
    def __init__(self, config: Configuration):
        super(Model, self).__init__()

        self.config = config
        self.sigma = config["sigma"]

        if config["autoencoder"]["type"] == "ours":
            self.ae = build_vqvae(
                config=config["autoencoder"],
                convert_to_sequence=True)
            self.ae.backbone.load_from_ckpt(config["autoencoder"]["ckpt_path"])
        else:
            if config["autoencoder"]["config"] == "f8":
                ae_config = vq_f8_ddconfig
            elif config["autoencoder"]["config"] == "f8_small":
                ae_config = vq_f8_small_ddconfig
            else:
                ae_config = vq_f16_ddconfig
            self.ae = VQModelInterface(ae_config, config["autoencoder"]["ckpt_path"])

        self.vector_field_regressor = build_vector_field_regressor(
            config=self.config["vector_field_regressor"])

    def load_from_ckpt(self, ckpt_path: str):
        loaded_state = torch.load(ckpt_path, map_location="cpu")

        is_ddp = False
        for k in loaded_state["model"]:
            if k.startswith("module"):
                is_ddp = True
                break
        if is_ddp:
            state = {k.replace("module.", ""): v for k, v in loaded_state["model"].items()}
        else:
            state = {f"module.{k}": v for k, v in loaded_state["model"].items()}

        dmodel = self.module if isinstance(self, torch.nn.parallel.DistributedDataParallel) else self
        dmodel.load_state_dict(state)

    def forward(
            self,
            observations: torch.Tensor,
            global_step: int=0) -> DictWrapper[str, Any]:
        """

        :param observations: [b, num_observations, num_channels, height, width]
        """

        batch_size = observations.size(0)
        num_observations = observations.size(1)
        assert num_observations > 2

        # Sample target frames and conditioning
        target_frames_indices = torch.randint(low=2, high=num_observations, size=[batch_size])
        target_frames = observations[torch.arange(batch_size), target_frames_indices]
        reference_frames_indices = target_frames_indices - 1
        reference_frames = observations[torch.arange(batch_size), reference_frames_indices]
        conditioning_frames_indices = torch.cat(
            [torch.randint(low=0, high=s - 1, size=[1]) for s in target_frames_indices], dim=0)
        conditioning_frames = observations[torch.arange(batch_size), conditioning_frames_indices]

        # Encode observations to latent codes
        with torch.no_grad():
            self.ae.eval()
            input_frames = torch.stack([target_frames, reference_frames, conditioning_frames], dim=1)
            if self.config["autoencoder"]["type"] == "ours":
                latents = self.ae(input_frames).latents
            else:
                flat_input_frames = rearrange(input_frames, "b n c h w -> (b n) c h w")
                flat_latents = self.ae.encode(flat_input_frames)
                latents = rearrange(flat_latents, "(b n) c h w -> b n c h w", n=3)
        target_latents = latents[:, 0]
        reference_latents = latents[:, 1]
        conditioning_latents = latents[:, 2]

        # Sample input latents
        noise = torch.randn_like(target_latents).to(target_latents.dtype).to(target_latents.device)
        timestamps = torch.rand(batch_size, 1, 1, 1).to(target_latents.dtype).to(target_latents.device)
        input_latents = (1 - (1 - self.sigma) * timestamps) * noise + timestamps * target_latents
    
        # Calculate target vectors
        target_vectors = (target_latents - (1 - self.sigma) * input_latents) / (1 - (1 - self.sigma) * timestamps)
        # Calculate time distances
        index_distances = (reference_frames_indices - conditioning_frames_indices).to(input_latents.device)

        
        # input_latents.requires_grad_(True)
        # Predict vectors
        reconstructed_vectors = self.vector_field_regressor(
            input_latents=input_latents,
            reference_latents=reference_latents,
            conditioning_latents=conditioning_latents,
            index_distances=index_distances,
            timestamps=timestamps.squeeze(3).squeeze(2).squeeze(1))

        if self.training and global_step < -1:
            dim = int(np.prod(target_latents.shape[1:]))
            # v = torch.randint_like(input_latents, 0, 2).float()*2 - 1
            v = torch.randn_like(input_latents)
            d_target_vectors_x = (
                - (1 - self.sigma) / (1 - (1 - self.sigma) * timestamps)
                ) * torch.ones_like(target_latents)
            sigma_t = 1 - (1 - self.sigma) * timestamps
            mu_t = timestamps
            d_logp = (mu_t * target_latents - input_latents) / sigma_t ** 2
            # reconstructed_vectors_fixed = reconstructed_vectors.detach()
            reconstructed_vectors_fixed = reconstructed_vectors
            right_vec_vp = (
                inner_prod(d_logp, v) * reconstructed_vectors_fixed.reshape((target_latents.shape[0], dim))
                - inner_prod(d_logp, v) * target_vectors.reshape((target_latents.shape[0], dim))
                - (d_target_vectors_x * v).reshape((target_latents.shape[0], dim))
            )
            # reconstructed_vectors_vjp = torch.autograd.grad(
            #     (reconstructed_vectors*v).sum(), 
            #     input_latents, 
            #     create_graph=True,
            #     retain_graph=True)[0].reshape((target_latents.shape[0], dim))
            
            h = 5e-6
            input_latents_tilde_p = input_latents + h * v
            reconstructed_vectors_tilde_p = self.vector_field_regressor(
                input_latents=input_latents_tilde_p,
                reference_latents=reference_latents,
                conditioning_latents=conditioning_latents,
                index_distances=index_distances,
                timestamps=timestamps.squeeze(3).squeeze(2).squeeze(1)
                )
            reconstructed_vectors_vjp = (
                (reconstructed_vectors_tilde_p - reconstructed_vectors) / h
                ).reshape((target_latents.shape[0], dim))


            est_diff_frob_vp = (reconstructed_vectors_vjp + right_vec_vp)
            est_diff_frob = est_diff_frob_vp * sigma_t.reshape((target_latents.shape[0], 1))
            est_div_err = torch.abs(est_diff_frob) / np.sqrt(float(dim))
            # est_div_err = (inner_prod(est_diff_frob, v) / float(dim))**2

        else:
            est_div_err = torch.zeros_like(reconstructed_vectors)

        return DictWrapper(
            # Inputs
            observations=observations,

            # Data for loss calculation
            reconstructed_vectors=reconstructed_vectors,
            target_vectors=target_vectors,
            est_diff_frob=est_div_err
            )

    @torch.no_grad()
    def generate_frames(
            self,
            observations: torch.Tensor,
            num_frames: int = None,
            steps: int = 100,
            warm_start: float = 0.0,
            past_horizon: int = -1,
            verbose: bool = False) -> torch.Tensor:
        """
        Generates num_frames frames conditioned on observations

        :param observations: [b, num_observations, num_channels, height, width]
        :param num_frames: number of frames to generate
        :param warm_start: part of the integration path to jump to
        :param steps: number of steps for sampling
        :param past_horizon: number of frames to condition on
        :param verbose: whether to display loading bar
        """

        # Encode observations to latents
        self.ae.eval()
        # record_dict = {}
        # record_dict['observations'] = observations.detach()
        if self.config["autoencoder"]["type"] == "ours":
            latents = self.ae(observations).latents
        else:
            flat_input_frames = rearrange(observations, "b n c h w -> (b n) c h w")
            flat_latents = self.ae.encode(flat_input_frames)
            latents = rearrange(flat_latents, "(b n) c h w -> b n c h w", n=observations.size(1))

        b, n, c, h, w = latents.shape
        if n == 1:
            latents = latents[:, [0, 0]]

        # Generate future latents
        # debug --------------------------
        gen = tqdm(range(num_frames), desc="Generating frames", disable=not verbose, leave=False)
        for _ in gen:
            def f(t: torch.Tensor, y: torch.Tensor):
                lower_bound = 0 if past_horizon == -1 else min(0, latents.size(1) - past_horizon)
                higher_bound = latents.size(1) - 1

                # Sample conditioning and reference
                conditioning_latents_indices = torch.randint(low=lower_bound, high=higher_bound, size=[b])
                conditioning_latents = latents[torch.arange(b), conditioning_latents_indices]
                reference_latents = latents[:, -1]

                # Calculate index distances
                index_distances = (higher_bound - conditioning_latents_indices).to(y.device)

                # Calculate vectors
                return self.vector_field_regressor(
                    input_latents=y,
                    reference_latents=reference_latents,
                    conditioning_latents=conditioning_latents,
                    index_distances=index_distances,
                    timestamps=t * torch.ones(b).to(latents.device))

            # Initialize with noise
            noise = torch.randn([b, c, h, w]).to(latents.device)
            y0 = (1 - (1 - self.sigma) * warm_start) * noise + warm_start * latents[:, -1]

            # Solve ODE
            next_latents = odeint(
                f,
                y0,
                t=torch.linspace(warm_start, 1, int((1 - warm_start) * steps)).to(y0.device),
                method="rk4"
               # method="dopri5"
            )[-1]
            latents = torch.cat([latents, next_latents.unsqueeze(1)], dim=1)

        # Close loading bar
        gen.close()
        if n == 1:
            latents = latents[:, 1:]
        # debug --------------------------

        # Decode to image space
        latents = rearrange(latents, "b n c h w -> (b n) c h w")
        if self.config["autoencoder"]["type"] == "ours":
            reconstructed_observations = self.ae.backbone.decode_from_latents(latents)
        else:
            reconstructed_observations = self.ae.decode(latents)
        reconstructed_observations = rearrange(reconstructed_observations, "(b n) c h w -> b n c h w", b=b)

        return reconstructed_observations
