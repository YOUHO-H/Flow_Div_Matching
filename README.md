# Improving Flow Matching by Aligning Flow Divergence
We use parts of code from [Lipman et al.,
2023](https://github.com/facebookresearch/flow_matching), [Davtyan et al., 2023](https://github.com/Araachie/river) [(Stark et al., 2024)](https://github.com/HannesStark/dirichlet-flow-matching) and [Finzi et al. (2023)](https://github.com/google-research/google-research/tree/9dda2b5e6503284eeb24e746d4103ed37019a80e/simulation_research/diffusion).


Conditional Flow Matching (CFM) is an efficient, simulation-free method for training flow-based generative models, but it struggles to accurately learn probability paths. We address this by introducing a new PDE-based error characterization and show that the total variation between learned and true paths can be bounded by combining the CFM loss with a divergence loss. This leads to a new training objective that jointly matches flow and divergence, significantly improving performance on tasks like dynamical systems, DNA sequence, and video generation—without sacrificing efficiency. 

To validate the efficacy and efficiency of the proposed FDM in enhancing FM across various bench-mark tasks, including 
## Density estimation on synthetic 2D data 
In this experiment, we train FM and FDM for sampling 2D synthetic checkboard data. See the [subdirectory README](https://github.com/Utah-Math-Data-Science/Flow_Div_Matching/blob/main/flow_matching_2d_Synthetic_FDM/README.md).


## DNA sequence generation

In this experiment, we demonstrate that FDM enhances FM with the conditional OT path and Dirichlet path (Stark et al.,2024) on the probability simplex for DNA sequence generation, both with and without guidance, following experiments
conducted in Stark et al., 2024. See the [subdirectory README](https://github.com/Utah-Math-Data-Science/Flow_Div_Matching/blob/main/dirichlet-flow-matching-FDM/README.md).

## Trajectory sampling for dynamical systems 
In this experiment, we compare FDM against FM and DM from (Finzi et al., 2023) on the Lorenz and FitzHugh-Nagumo models (Farazmand & Sapsis, 2019). See the [subdirectory README](https://github.com/Utah-Math-Data-Science/Flow_Div_Matching/blob/main/Aligning-Flow-Div-User-Defined-Sampling/README.rst).

## Video prediction via latent FM

We train a latent FM (Davtyan et al., 2023) and a latent FDM for video prediction. We further utilize a pre-trained VQGAN (Esser et al., 2021) to encode (resp. decode) each frame of the video to (resp. from) the latent space. See the [subdirectory README](https://github.com/Utah-Math-Data-Science/Flow_Div_Matching/blob/main/LFM-FDM-KTH/README.md).



