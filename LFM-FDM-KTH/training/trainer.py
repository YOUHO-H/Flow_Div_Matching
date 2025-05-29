import os
from typing import Any, Tuple
import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_polynomial_decay_schedule_with_warmup
from PIL import Image
from evaluation.evaluator import Evaluator
from lutils.configuration import Configuration
from lutils.constants import MAIN_PROCESS
from lutils.dict_wrapper import DictWrapper
from lutils.logger import Logger
from lutils.logging import to_video, make_observations_grid
from training.utils import check_ddp_consistency
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import torch.nn.functional as F
import wandb
from collections import OrderedDict

class I3D(nn.Module):
    def __init__(self):
        super().__init__()
        # https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1
        self.detector = torch.jit.load('./i3d_torchscript.pt').eval()
        self.detector_args = dict(rescale=False, resize=True, return_features=True)

    def forward(self, x, target):  # perceptual loss
        with torch.no_grad():
            target = self.detector(target, **self.detector_args)  # N, 400
        x = self.detector(x, **self.detector_args)
        return F.mse_loss(x, target)

    @staticmethod
    def compute_stats(feats: np.ndarray):
        feats = feats.astype(np.float64)
        mu = feats.mean(axis=0)  # [d]
        sigma = np.cov(feats, rowvar=False)  # [d, d]
        return mu, sigma

    @torch.no_grad()
    def fvd(self, feats_fake, feats_real):
        mu_gen, sigma_gen = self.compute_stats(feats_fake)
        mu_real, sigma_real = self.compute_stats(feats_real)
        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
        fvd = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return float(fvd)

class Trainer:
    """
    Class that handles the training
    """

    def __init__(
            self,
            rank: int,
            run_name: str,
            config: Configuration,
            dataset: Dataset,
            sampler: torch.utils.data.distributed.Sampler,
            num_gpus: int,
            device: torch.device,
            lambda_reg: float = 0.0):
        """
        Initializes the Trainer

        :param rank: rank of the current process
        :param config: training configuration
        :param dataset: dataset to train on
        :param sampler: sampler to create the dataloader with
        :param device: device to use for training
        """
        super(Trainer, self).__init__()

        self.config = config
        self.rank = rank
        self.is_main_process = self.rank == MAIN_PROCESS
        self.num_gpus = num_gpus
        self.device = device
        self.reg = lambda_reg
        # Create folder for saving
        self.run_path = os.path.join("runs", run_name)
        os.makedirs(self.run_path, exist_ok=True)
        os.makedirs(os.path.join(self.run_path, "checkpoints"), exist_ok=True)

        # Setup dataloader
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config["batching"]["batch_size"],
            shuffle=False,
            num_workers=self.config["batching"]["num_workers"],
            sampler=sampler,
            pin_memory=True)

        # Setup losses
        self.flow_matching_loss = nn.MSELoss()

        # Optimizer will be defined in train_epoch
        self.optimizer = None

        # Scheduler will be defined in train_epoch
        self.lr_scheduler = None

        self.global_step = 0

        self.i3d_model = I3D().to(self.device)

    def init_optimizer(self, model: nn.Module):
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config["optimizer"]["learning_rate"],
            weight_decay=self.config["optimizer"]["weight_decay"],
            betas=(0.9, 0.999))
        self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.config["optimizer"]["num_warmup_steps"],
            num_training_steps=self.config["optimizer"]["num_training_steps"],
            power=0.75)

    def get_lr(self):
        assert self.optimizer is not None

        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train(
            self,
            model: nn.Module,
            logger: Logger,
            evaluator: Evaluator,
            test_only: bool = False,
            scalar_logging_frequency: int = 1000,
            media_logging_frequency: int = 5000,
            saving_frequency: int = 5000,
            evaluation_frequency: int = 10000,
            checkpointing_frequency: int = 20000,
            ):
        """
        Trains the model for one epoch

        """
        model.train()
        dmodel = model if not isinstance(model, nn.parallel.DistributedDataParallel) else model.module

        # Setup optimizer and scheduler if not yet
        if self.optimizer is None:
            self.init_optimizer(model)

        # Setup loading bar
        train_gen = tqdm(self.dataloader, desc="Batches", disable=not self.is_main_process, leave=False)
        train_loss = 0
        train_der = 0
        batch_count = 0
        best_fvd = 1e12
    
        # Evaluate the model
        if test_only:

            path_media_out_ = os.path.join(self.config['out_path'], 'log_media')
            os.makedirs(path_media_out_, exist_ok=True)
            for k in [299999]:
                if '_b' in self.config['out_path'].split('/')[-1]:
                    chkp_path = os.path.join(self.config['out_path'], f'model.pth')
                    path_media_out = os.path.join(path_media_out_, str(k))
                    checkpoint = torch.load(chkp_path)['model']
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint.items():
                        name = k.replace("module.", "")  # Remove `module.`
                        new_state_dict[name] = v
                    model.load_state_dict(new_state_dict)
                else:
                    chkp_path = os.path.join(self.config['out_path'], f'ckpt_{k}.pth')
                    path_media_out = os.path.join(path_media_out_, str(k))
                    os.makedirs(path_media_out, exist_ok=True)
                    loaded_state = torch.load(chkp_path)
                    model.load_state_dict(loaded_state["model"])
                test_fvd, _ = evaluator.evaluate(model=model, logger=logger, 
                                                global_step=k, 
                                                max_num_batches=99,
                                                path_media_out=path_media_out, 
                                                i3d=self.i3d_model, 
                                                device=self.device)
                if test_fvd < best_fvd:
                    best_fvd = test_fvd
                wandb.log({'FVD': test_fvd,
                        'Best FVD': best_fvd,
                        'iter': k})

        else:
            chkp_path = '/root/workspace/A_out/river/run_kth_13/ckpt_299999.pth'
            loaded_state = torch.load(chkp_path)
            model.load_state_dict(loaded_state["model"])

            for batch in train_gen:

                # Fetch data
                observations = batch.cuda()
                batch_size = observations.size(0)
                num_observations = self.config["num_observations"]
                observations = observations[:, :num_observations]

                # Zero the gradients
                self.optimizer.zero_grad()
                # Forward the model
                model_outputs = model(observations, self.global_step)

                # Compute the loss
                loss, auxiliary_output = self.calculate_loss(model_outputs)
                loss_der = model_outputs.est_diff_frob.mean()

                train_loss = train_loss + loss.item()
                train_der = train_der + loss_der.item()
                batch_count += 1
                # Backward pass
                loss = loss + loss_der * self.reg
                loss.backward()

                # Uncomment this if nans appear in the loss during the training
                self.reduce_gradients(model, self.num_gpus)

                # Optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()

                # Log scalars
                
                if self.global_step % scalar_logging_frequency == 0 and self.global_step > 0 and self.is_main_process:
                    lr = self.get_lr()
                    print('Training Loss is : {:.6f} | Training diff : {:.6f} | reg: {}'.format(train_loss/batch_count, train_der/batch_count, self.reg))
                    wandb.log({'Training Loss': train_loss/batch_count,
                               'Training diff': train_der/batch_count,
                               'Learning Rate': lr,
                               'reg': self.reg,})
                    train_loss = 0
                    train_der = 0
                    batch_count = 0

                # Finalize logs
                logger.finalize_logs(step=self.global_step)

                if self.global_step % evaluation_frequency == 9999 and self.global_step>49999:
                    self.save_checkpoint(model, f"{self.config['out_path']}/ckpt_{self.global_step}.pth")
                self.global_step += 1

            # Close loading bar
            train_gen.close()


    def calculate_loss(
            self,
            results: DictWrapper[str, Any]) -> Tuple[torch.Tensor, DictWrapper[str, Any]]:
        """
        Calculates the loss

        :param results: Dict with the model outputs
        :return: [1,] The loss value
        """

        # Flow matching loss
        flow_matching_loss = self.flow_matching_loss(
            results.reconstructed_vectors,
            results.target_vectors)

        # Sum up all the losses
        loss_weights = self.config["loss_weights"]
        loss = loss_weights["flow_matching_loss"] * flow_matching_loss

        # DDP hack
        def add_zero_to_loss(value):
            if v is None:
                return loss
            return loss + value.mul(0).mean()

        for _, v in results.items():
            if isinstance(v, list):
                for ev in v:
                    loss = add_zero_to_loss(ev)
            else:
                loss = add_zero_to_loss(v)

        # Create auxiliary output
        auxiliary_output = DictWrapper(
            # Total loss
            total_loss=loss,

            # Loss terms
            flow_matching_loss=flow_matching_loss
        )

        return loss, auxiliary_output

    def log_scalars(self, loss_terms: DictWrapper[str, Any], other_data: DictWrapper[str, Any], logger: Logger):
        for k, v in loss_terms.items():
            logger.log(f"Training/Loss/{k}", v)

        # Log training stats
        logger.log(f"Training/Stats/learning_rate", self.get_lr())
        logger.log(f"Training/Stats/total_loss_is_nan", torch.isnan(loss_terms.total_loss).to(torch.int8))
        logger.log(f"Training/Stats/total_loss_is_inf", torch.isinf(loss_terms.total_loss).to(torch.int8))

        # Other stats
        for k, v in other_data.items():
            logger.log(f"Training/Stats/{k}", v)

    @staticmethod
    def log_media(results: DictWrapper[str, Any], logger: Logger, path_media_out: str, i3d: None, device: None):
        # seq num
        num_sequences = results.observations.size(0)

        # Log images grid
        grid = make_observations_grid(
            [
                results.observations,
                results.generated_observations,
            ],
            num_sequences=num_sequences)
        # logger.log(f"Training/Media/reconstructed_observations", logger.wandb().Image(grid))
        
        # Log real videos
        real_videos = to_video(results.observations[:num_sequences])
        # logger.log("Training/Media/real_videos", logger.wandb().Video(real_videos, fps=7))

        # Log generated videos
        generated_videos = to_video(results.generated_observations)
        num_videos = generated_videos.shape[0]
        for k in range(num_videos):
            v_arr = generated_videos[k].transpose(0, 2, 3, 1)
            fps=1
            clip = ImageSequenceClip(list(v_arr), fps=fps)
            clip.write_videofile(os.path.join(path_media_out, f'generated_video_{k}.mp4'), fps=fps)

            v_arr_real = real_videos[k].transpose(0, 2, 3, 1)
            clip_real = ImageSequenceClip(list(v_arr_real), fps=fps)
            clip_real.write_videofile(os.path.join(path_media_out, f'real_video_{k}.mp4'), fps=fps)

            if k > 4:
                break

        real_videos_T = real_videos.transpose(0,2,1,3,4)
        
        real_videos_T = torch.from_numpy(real_videos_T).to(torch.float32).contiguous().to(device) / 256
        feats_real = i3d.detector(real_videos_T, **i3d.detector_args).detach().cpu().numpy()

        generated_videos_T = generated_videos.transpose(0,2,1,3,4)
        generated_videos_T = torch.from_numpy(generated_videos_T).to(torch.float32).contiguous().to(device) / 256
        feats_fake = i3d.detector(generated_videos_T, **i3d.detector_args).detach().cpu().numpy()
        
        fvd_score = i3d.fvd(feats_fake, feats_real) / num_sequences

        return fvd_score
        # logger.log("Training/Media/generated_videos", logger.wandb().Video(generated_videos, fps=7))

    @staticmethod
    def reduce_gradients(model: nn.Module, num_gpus: int):
        params = [param for param in model.parameters() if param.grad is not None]
        if len(params) > 0:
            flat = torch.cat([param.grad.flatten() for param in params])
            if num_gpus > 1:
                torch.distributed.all_reduce(flat)
                flat /= num_gpus
            torch.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
            grads = flat.split([param.numel() for param in params])
            for param, grad in zip(params, grads):
                param.grad = grad.reshape(param.shape)

    def save_checkpoint(self, model: nn.Module, checkpoint_name: str = None):
        if self.num_gpus > 1:
            check_ddp_consistency(model, r".*\..+_(mean|var|tracked)")

        if self.is_main_process:
            state_dict = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "model": model.state_dict(),
                "global_step": self.global_step
            }
            if checkpoint_name:
                torch.save(state_dict, os.path.join(self.run_path, "checkpoints", checkpoint_name))
            torch.save(state_dict, os.path.join(self.run_path, "checkpoints", "latest.pth"))

    def load_checkpoint(self, model: nn.Module, checkpoint_name: str = None):
        if checkpoint_name is None:
            checkpoint_name = "latest.pth"
        filename = os.path.join(self.run_path, "checkpoints", checkpoint_name)
        if not os.path.isfile(filename):
            raise Exception(f"Cannot load model: no checkpoint found at '{filename}'")

        # Init optimizer and scheduler if not yet
        if self.optimizer is None:
            self.init_optimizer(model)

        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        loaded_state = torch.load(filename, map_location=map_location)
        self.optimizer.load_state_dict(loaded_state["optimizer"])
        self.lr_scheduler.load_state_dict(loaded_state["lr_scheduler"])

        is_ddp = False
        for k in loaded_state["model"]:
            if k.startswith("module"):
                is_ddp = True
                break
        if is_ddp:
            state = {k.replace("module.", ""): v for k, v in loaded_state["model"].items()}
        else:
            state = {f"module.{k}": v for k, v in loaded_state["model"].items()}

        dmodel = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        dmodel.load_state_dict(state)

        self.global_step = loaded_state["global_step"]
