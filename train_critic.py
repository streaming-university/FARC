import os
import json
import utils
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from FARC import FARCritic

BS = 256
T = 2000
obs_dim = 150
hidden_size = 128
epochs = 100

data_dir = "./Data/testbed_dataset/train/"


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic = FARCritic(input_dim=obs_dim).to(device)

    # optimizers
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    loss_fn = torch.nn.MSELoss()

    trace_files = glob(os.path.join(data_dir, '*.json'), recursive=True)

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        critic.train()

        for filename in tqdm(trace_files):
            with open(filename, "r") as file:
                call_data = json.load(file)

            optimizer.zero_grad()
            observations = np.asarray(call_data['observations'], dtype=np.float32)
            bandwidth_predictions = np.asarray(call_data['bandwidth_predictions'], dtype=np.float32)
            # solve NaN values
            video_quality = utils.forward_fill(np.asarray(call_data['video_quality'], dtype=np.float32))
            audio_quality = utils.forward_fill(np.asarray(call_data['audio_quality'], dtype=np.float32))

            batched_observations = utils.prepare_batches(observations, BS)
            batched_predictions = utils.prepare_batches(bandwidth_predictions, BS)
            batched_vid_qualities = utils.prepare_batches(video_quality, BS)
            batched_aud_qualities = utils.prepare_batches(audio_quality, BS)

            for batch_obs, batch_preds, batch_vid, batch_aud in zip(batched_observations, batched_predictions,
                                                                    batched_vid_qualities, batched_aud_qualities):
                optimizer.zero_grad()
                state = torch.tensor(batch_obs, dtype=torch.float32).reshape(-1, 1, obs_dim).to(device)
                action = torch.tensor(batch_preds, dtype=torch.float32).unsqueeze(0).reshape(-1, 1).to(device)
                vid = torch.tensor(batch_vid, dtype=torch.float32).unsqueeze(0).reshape(-1, 1).to(device)
                aud = torch.tensor(batch_aud, dtype=torch.float32).unsqueeze(0).reshape(-1, 1).to(device)


                critic_estimate = critic(state, action)

                loss = loss_fn(critic_estimate, torch.cat((vid, aud), dim=1))
                if torch.isnan(loss):
                    continue
                loss.backward()
                optimizer.step()

        scheduler.step()
        if epoch % 10 == 0:
            utils.save_checkpoint(critic, optimizer, epoch,
                                  filename="./model_checkpoints/critic_{}_{}.pth".format(critic.name, epoch))


if __name__ == '__main__':
    train()
