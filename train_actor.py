import os
import json
import utils
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from FARC import FARCtor, FARCritic

BS = 256
T = 2000
obs_dim = 150
hidden_size = 128
epochs = 100

# discount factor for true capacity reward
F_CONSERVE = 0.96

data_dir = "./Data/emulated_dataset/train/"


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = FARCtor(input_size=obs_dim, hidden_size=hidden_size).to(device)
    critic = FARCritic(input_dim=obs_dim).to(device)
    # load the final critic model
    critic = utils.load_checkpoint(critic, "./model_checkpoints/critic_best.pth")

    # optimizers
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    loss_fn = torch.nn.MSELoss()

    trace_files = glob(os.path.join(data_dir, '*.json'), recursive=True)

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        actor.train()
        critic.zero_grad()

        for filename in tqdm(trace_files):
            with open(filename, "r") as file:
                call_data = json.load(file)

            optimizer.zero_grad()
            observations = np.asarray(call_data['observations'], dtype=np.float32)
            true_capacities = np.asarray(call_data['true_capacity'], dtype=np.float32)
            batched_observations = utils.prepare_batches(observations, BS)
            batched_capacities = utils.prepare_batches(true_capacities, BS)

            for batch_obs, batch_caps in zip(batched_observations, batched_capacities):
                optimizer.zero_grad()

                state = torch.tensor(batch_obs, dtype=torch.float32).reshape(-1, 1, obs_dim).to(device)
                target = torch.tensor(batch_caps, dtype=torch.float32).unsqueeze(0).reshape(-1, 1).to(device)

                h = torch.zeros((BS, hidden_size))
                c = torch.zeros((BS, hidden_size))

                action, h, c = actor(state, h, c)
                bw_prediction = action[:, :, 0]
                # gets the critic reward for the actor prediction
                actor_reward = critic(state, bw_prediction)
                # gets the discounted critic reward for the true capacity
                max_reward = critic(state, target[:, :] * F_CONSERVE)

                loss = loss_fn(actor_reward, max_reward)
                if torch.isnan(loss):
                    continue
                loss.backward()
                optimizer.step()

        scheduler.step()
        if epoch % 10 == 0:
            print("Saving model...")
            utils.save_checkpoint(actor, optimizer, epoch,
                                  filename="./model_checkpoints/actor_{}_{}.pth".format(actor.name, epoch))
            utils.save_onnx_model(actor, actor.name + "_{}".format(epoch), BS, T, obs_dim, hidden_size)


if __name__ == '__main__':
    train()
