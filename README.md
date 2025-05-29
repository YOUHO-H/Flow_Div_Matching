# Improving Flow Matching by Aligning Flow Divergence

Conditional Flow Matching (CFM) is an efficient, simulation-free method for training flow-based generative models, but it struggles to accurately learn probability paths. We address this by introducing a new PDE-based error characterization and show that the total variation between learned and true paths can be bounded by combining the CFM loss with a divergence loss. This leads to a new training objective that jointly matches flow and divergence, significantly improving performance on tasks like dynamical systems, DNA sequence, and video generation—without sacrificing efficiency.

To validate the efficacy and efficiency of the proposed FDM in enhancing FM across various bench-mark tasks, including 
## Density estimation on synthetic 2D data 

## DNA sequence generation

## Trajectory sampling for dynamical systems 


## Video prediction via latent FM



