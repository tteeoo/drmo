import os 
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor

def get_images():
    image_values = {}
    data = [[], []]
    opened, closed = './data/open/', './data/closed/'
    for x in os.listdir(opened):
        image_values[opened+'/'+x] = True 
    for x in os.listdir(closed):
        image_values[closed+'/'+x] = False
    
    for x in image_values:
        data[0].append(True) if image_values[x] else data[0].append(False)
        data[1].append(x)

    return data

class EyeNet(nn.Module):
    """ Class representing the neural network. """

    def __init__(self):
        """ Initialize the model with appropriate starting weights. """

        super(EyeNet, self).__init__()
        self.pool = nn.MaxPool2d(1, 1)
        self.conv1 = nn.Conv2d(3, 48, 48)
        self.conv2 = nn.Conv2d(48, 16, 1)
        self.fc1 = nn.Linear(16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """ Calculations based on image tensor input. """

        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(-1, 16)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)

        return x

net = EyeNet()
device = 'cpu'
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

class ImageData(Dataset):
    """ Class to represent image and label input. """

    def __init__(self, yx, width=48, height=48, transform=None):
        self.width = width
        self.height = height
        self.transform = transform
        y, x = yx 
        self.y = y # array of label
        self.x = x # array of image paths

    def __getitem__(self, index):
        """ Process image and labels to be sent to the NN. """

        img = cv2.cvtColor(self.x[index], cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((self.width, self.height)) 
        img = img.convert('RGB') #convert image to RGB channel
        img = np.asarray(img).transpose(-1, 0, 1) # we have to change the dimensions from width x height x channel (WHC) to channel x width x height (CWH)
        img = img/255
        img = torch.from_numpy(np.asarray(img)) # create the image tensor
        label = torch.from_numpy(np.asarray(self.y[index]).reshape([1, 1])) # create the label tensor
        
        return img, label, self.x[index]
    
    def __len__(self):
        return len(self.x)

dset = DataLoader(ImageData(get_images()), batch_size=16, shuffle=True, num_workers=1)

# set up the NN
net.to(device)
if os.path.isfile('./data/net.pth'):
    net.load_state_dict(torch.load('./data/net.pth', map_location=device))

def classify(img):
    """ Runs the given image through the model and returns the results. """

    data = ImageData([[True], [img]])
    tensor_image, _, img_path = data[0] # get image tensor etc
    tensor_image = tensor_image.to(device).unsqueeze(0) # format image tensor
    
    output = net(tensor_image.float()) # run image tensor through network to get predicted value (1 is cat 0 is not)

    _, predicted = torch.max(output, 1)
    cat = True if predicted.item() == 1 else False

    return cat, img_path

def train(epochs, start_from_scratch=False):
    """ Train the NN on the dataset in cats/. """
        
    n = EyeNet() if start_from_scratch else net
    n.to(device)

    for epoch in range(epochs):
        for data in dset:

            inputs, labels, _ = data
            inputs, labels = inputs.to(device), labels.to(device).squeeze().long()

            optimizer.zero_grad()

            outputs = n(inputs.float()) # predicted values (1 if cat 0 if not cat)
            loss = criterion(outputs, labels) # calculate the loss
            print('epoch:', epoch, 'loss of batch:', loss.item())
            loss.backward() # calculate improved weights based on loss
            optimizer.step() # optimize with new weights

    print('Finished Training')
    torch.save(net.state_dict(), './data/net.pth')
