# Dirichlet Flow Matching by Aligning Flow Divergence

### Conda environment
```yaml
conda create -c conda-forge -n seq python=3.9
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
pip install torch_geometric jupyterlab gpustat pyyaml wandb biopython spyrmsd einops biopandas plotly seaborn prody tqdm lightning imageio tmtools "fair-esm[esmfold]" e3nn
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cu113.htm

# The 'selene' libraries below are required for the promoter design experiments
git clone https://github.com/kathyxchen/selene.git
cd selene
git checkout custom_target_support
python setup.py build_ext --inplace
python setup.py install

pip install pyBigWig pytabix cooler pyranges biopython cooltools
```

## Toy experiments
The commands below are for linear flow matching and dirichlet flow matching. K in the paper corresponds to `--toy_simplex_dim` here. <LAMBDA> here is the hyper-paremeter for the divergence loss.
```yaml
python train_dna.py --run_name trainToy_linear_dim40 --dataset_type toy_sampled --limit_val_batches 1000 --toy_seq_len 4 --toy_simplex_dim 40 --toy_num_cls 1 --val_check_interval 5000 --batch_size 512 --print_freq 100 --wandb --model cnn --mode riemannian --prob_reg <LAMBDA>

python train_dna.py --run_name trainToy_diri_dim40 --dataset_type toy_sampled --limit_val_batches 1000 --toy_seq_len 4 --toy_simplex_dim 40 --toy_num_cls 1 --val_check_interval 5000 --batch_size 512 --print_freq 100 --wandb --model cnn --prob_reg <LAMBDA>

```

## Promoter design experiments

Download the dataset from https://zenodo.org/records/7943307 and place it in `data`.

Example command for training. <LAMBDA> here is the hyper-paremeter for the divergence loss.

```yaml
python train_promo.py --run_name train_dirichlet_fm --batch_size 128 --wandb --num_workers 4 --check_val_every_n_epoch 5 --num_integration_steps 100 --limit_val_batches 16 --prob_reg <LAMBDA>
```

