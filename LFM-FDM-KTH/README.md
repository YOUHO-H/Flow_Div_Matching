# Latent Flow Matching for KTH Video Prediction with Divergence Matching
  
## Prerequisites

For convenience, we provide an `environment.yml` file that can be used to install the required packages 
to a `conda` environment with the following command 

```conda env create -f environment.yml```

The code was tested with cuda=12.1 and python=3.9.

## Pretrained VQGAN for KTH

Go https://huggingface.co/cvg-unibe/river_kth_64/blob/main/vqvae.ckpt for downloading the pretrained VQGAN for KTH dataset


## Datasets

Download the videos from the [dataset's official website](https://www.csc.kth.se/cvap/actions/).

The training code expects the dataset to be packed into .hdf5 files in a custom manner. 
To create such files, use the provided `dataset/convert_to_h5.py` script. 
Usage example:

```angular2html
python dataset/convert_to_h5.py --out_dir <directory_to_store_the_dataset> --data_dir <path_to_video_frames> --image_size 128 --extension png
```

The output of `python dataset/convert_to_h5.py --help` is as follows:

The video frames at `--data_dir` should be organized in the following way:

```angular2html
data_dir/
|---train/
|   |---00000/
|   |   |---00000.png
|   |   |---00001.png
|   |   |---00002.png
|   |   |---...
|   |---00001/
|   |   |---00000.png
|   |   |---00001.png
|   |   |---00002.png
|   |   |---...
|   |---...
|---val/
|   |---...
|---test/
|   |---...
```


### Training main model

To launch the training of the main model, use the `train.py` script from this repository.
Usage example:

```angular2html
python train.py --config <path_to_config> --run-name <run_name> --wandb --FDM <LAMBDA>
```

where <LAMBDA> is the hyper-parameter for divergence matching.
