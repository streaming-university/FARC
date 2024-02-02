import os
import torch
import numpy as np


def forward_fill(data):
    # forward fill NaN values with the first next non-NaN value
    for i in range(len(data) - 1):
        if np.isnan(data[i]):
            # find the next non-NaN value
            for j in range(i + 1, len(data)):
                if not np.isnan(data[j]):
                    data[i] = data[j]
                    break
    return data


def save_onnx_model(model, name, BS, T, obs_dim, hidden_size):
    save_path = "./onnx_model/{}.onnx".format(name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.to("cpu")
    model.eval()

    # prepare dummy inputs: 1 episode x T timesteps x obs_dim features
    dummy_inputs = np.asarray(np.random.uniform(0, 1, size=(BS, T, obs_dim)), dtype=np.float32)
    torch_dummy_inputs = torch.as_tensor(dummy_inputs)
    torch_initial_hidden_state, torch_initial_cell_state = torch.as_tensor(
        np.zeros((1, hidden_size), dtype=np.float32)), torch.as_tensor(np.zeros((1, hidden_size), dtype=np.float32))

    torch.onnx.export(
        model, (torch_dummy_inputs[0:1, 0:1, :], torch_initial_hidden_state, torch_initial_cell_state), save_path,
        opset_version=11,
        input_names=['obs', 'hidden_states', 'cell_states'],  # the model's input names
        output_names=['output', 'state_out', 'cell_out'],  # the model's output names
    )


def save_checkpoint(model, optimizer, epoch, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, filename, optimizer=None):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    return model


def prepare_batches(data, batch_size):
    # calculate the number of full batches and only yield them
    num_batches = len(data) // batch_size

    for i in range(num_batches - 1):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        yield data[start_idx:end_idx]
