from typing import Any
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
from lutils.configuration import Configuration
from lutils.constants import MAIN_PROCESS
from lutils.dict_wrapper import DictWrapper
from lutils.logger import Logger
from lutils.logging import to_video, make_observations_grid
from lutils.running_average import RunningMean
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

class Evaluator:
    """
    Class that handles the evaluation
    """

    def __init__(
            self,
            rank: int,
            config: Configuration,
            dataset: Dataset,
            device: torch.device):
        """
        Initializes the Trainer

        :param rank: rank of the current process
        :param config: training configuration
        :param dataset: dataset to train on
        :param sampler: sampler to create the dataloader with
        :param device: device to use for training
        """
        super(Evaluator, self).__init__()

        self.config = config
        self.rank = rank
        self.is_main_process = self.rank == MAIN_PROCESS
        self.device = device

        # Setup dataloader
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config["batching"]["batch_size"],
            shuffle=True,
            num_workers=self.config["batching"]["num_workers"],
            pin_memory=True)

        # Setup losses
        self.flow_matching_loss = nn.MSELoss()

        self.running_means = RunningMean()

    @torch.no_grad()
    def evaluate(
            self,
            model: nn.Module,
            logger: Logger,
            global_step: int,
            max_num_batches: int,
            path_media_out: str, 
            i3d: None, 
            device: None):
        """
        Evaluates the model

        """

        if not self.is_main_process:
            return

        model.eval()

        # Setup loading bar
        fvd_sum = 0
        fvd_count = 0
        eval_gen = tqdm(
            self.dataloader,
            total=min(max_num_batches, len(self.dataloader)),
            desc="Evaluation: Batches",
            disable=not self.is_main_process,
            leave=False)

        worst_batch = None
        worst_batch_fvd = 0
        for i, batch in enumerate(eval_gen):
            if i >= max_num_batches:
                break

            # Fetch data
            observations = batch.cuda()
            num_observations = self.config["num_observations"]
            observations = observations[:, :num_observations]

            # Forward the model
            model_outputs = model(
                observations)

            # Compute the loss
            loss_output = self.calculate_loss(model_outputs)

            # Accumulate scalars
            self.running_means.update(loss_output)

            is_log_media = True
            # Log media
            dmodel = model if not isinstance(model, nn.parallel.DistributedDataParallel) else model.module
            model_outputs["generated_observations"] = dmodel.generate_frames(
                observations=observations[:observations.size(0), :self.config["condition_frames"]],
                num_frames=self.config["frames_to_generate"],
                steps = 100,
                verbose=self.is_main_process)
            fvd = self.log_media(model_outputs, 
                                 logger, 
                                 global_step,
                                 i,
                                 is_log_media,
                                 path_media_out, 
                                 i3d, device)
            if fvd > worst_batch_fvd:
                worst_batch_fvd = fvd
                worst_batch = i
            print('* Batch FVD: {:.2f} | Worst Batch / FVD {} / {:.2f}'.format(fvd, worst_batch, worst_batch_fvd))
            fvd_sum += fvd
            fvd_count += 1
            if i == 12:
                batch12_fvd = fvd
        
        fvd_est = fvd_sum / fvd_count
        print(" ------------ FVD: {:.2f} on {} Val Batches ------------".format(fvd_est, fvd_count))

        # Log scalars
        for k, v in self.running_means.get_values().items():
            logger.log(f"Validation/Loss/{k}", v)

        # Finalize logs
        logger.finalize_logs(step=global_step)

        # Close loading bar
        eval_gen.close()

        # Reset the model to train
        model.train()

        return fvd_est, batch12_fvd

    @torch.no_grad()
    def calculate_loss(
            self,
            results: DictWrapper[str, Any]) -> DictWrapper[str, Any]:
        """
        Calculates the loss

        :param results: Dict with the model outputs
        :return: [1,] The loss value
        """

        # Flow matching loss
        flow_matching_loss = self.flow_matching_loss(
            results.reconstructed_vectors,
            results.target_vectors)

        # Create auxiliary output
        output = DictWrapper(
            # Loss terms
            flow_matching_loss=flow_matching_loss
        )

        return output

    @staticmethod
    def log_media(results: DictWrapper[str, Any], 
                  logger: Logger, 
                  global_step: int,
                  batch_id: int,
                  log_media: bool,
                  path_media_out: str, 
                  i3d: None, 
                  device: None):
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
        fps = 2
        if log_media:
            resize_width = 160
            resize_height = 120
            batch_path_media_out = os.path.join(path_media_out, f'step_{global_step}_batch_{batch_id}')
            os.makedirs(batch_path_media_out, exist_ok=True)
            for k in range(num_videos):
                v_arr = generated_videos[k].transpose(0, 2, 3, 1)
                resized_frames = [cv2.resize(frame, (resize_width, resize_height)) for frame in v_arr]
                clip = ImageSequenceClip(resized_frames, fps=fps)
                clip.write_videofile(os.path.join(batch_path_media_out, f'generated_video_{batch_id}_{k}.mp4'), fps=fps)

        real_videos_T = real_videos.transpose(0,2,1,3,4)
        real_videos_T = torch.from_numpy(real_videos_T).to(torch.float32).contiguous().to(device) / 256
        real_videos_T = real_videos_T[:, :, ::fps, :, :]
        feats_real = i3d.detector(real_videos_T, **i3d.detector_args).detach().cpu().numpy()

        generated_videos_T = generated_videos.transpose(0,2,1,3,4)
        generated_videos_T = torch.from_numpy(generated_videos_T).to(torch.float32).contiguous().to(device) / 256
        generated_videos_T = generated_videos_T[:, :, ::fps, :, :]
        feats_fake = i3d.detector(generated_videos_T, **i3d.detector_args).detach().cpu().numpy()
    
        fvd_score = i3d.fvd(feats_fake, feats_real) 

        return fvd_score