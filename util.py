import os 
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data-drmo/'))

def get_images():
    """Returns a list of labelled image paths for neural network training."""

    data = ([], [])
    opened, closed = os.path.join(data_path, 'open/'), os.path.join(data_path, 'closed/')

    for f in os.listdir(opened):
        data[0].append(True)
        data[1].append(os.path.join(opened, f))

    for f in os.listdir(closed):
        data[0].append(False)
        data[1].append(os.path.join(closed, f))

    return data
