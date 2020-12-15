import os 
import util
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from data import ImageData
from torch.utils.data import DataLoader

class EyeNet(nn.Module):
    """Class representing the neural network."""

    def __init__(self):
        """Initialize the model's layers."""

        super(EyeNet, self).__init__()
        self.pool = nn.MaxPool2d(1, 1)
        self.conv1 = nn.Conv2d(3, 48, 48)
        self.conv2 = nn.Conv2d(48, 16, 1)
        self.fc1 = nn.Linear(16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """Calculate a forward pass through the layers."""

        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(-1, 16)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)

        return x

net = EyeNet()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

dset = DataLoader(ImageData(util.get_images()), batch_size=16, shuffle=True, num_workers=1)

# Set up the NN
net.to(util.device)
if os.path.isfile(os.path.join(util.data_path, 'net.pth')):
    net.load_state_dict(torch.load(os.path.join(util.data_path, 'net.pth'), map_location=util.device))

def classify(img):
    """Runs the given image through the model and returns the results."""

    data = ImageData([[True], [img]])
    tensor_image, _, img_path = data[0] # get image tensor etc
    tensor_image = tensor_image.to(util.device).unsqueeze(0) # format image tensor
    output = net(tensor_image.float()) # forward pass

    _, predicted = torch.max(output, 1)
    opened = True if predicted.item() == 1 else False

    return opened

def train(epochs):
    """Train the model on the data."""
        
    for epoch in range(epochs):
        for data in dset:

            inputs, labels, _ = data
            inputs, labels = inputs.to(util.device), labels.to(util.device).squeeze().long()

            optimizer.zero_grad()

            outputs = net(inputs.float()) # predicted values (1 if cat 0 if not cat)
            loss = criterion(outputs, labels) # calculate the loss
            print('epoch:', epoch, 'loss of batch:', loss.item())
            loss.backward() # calculate improved weights based on loss
            optimizer.step() # optimize with new weights

    print('Finished Training')
    torch.save(net.state_dict(), os.path.join(util.data_path, 'net.pth'))
    print('Saved model to', os.path.join(util.data_path, 'net.pth'))

