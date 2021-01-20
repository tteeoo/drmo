import os 
import sys
import time
import requests

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

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

class FileManager:
    """Class to represent the various data files."""
    
    def __init__(self):
        self.paths = {os.path.join(data_path, x.split('/')[-1]): x for x in (
            'https://directory.theohenson.com/file/net.pth', 
            'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
            'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_lefteye_2splits.xml',
            'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_righteye_2splits.xml',
            'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
        )}
        self.fully_installed = True
        for p in self.paths:
            if not os.path.isfile(p):
                self.fully_installed = False

    def install(self):
        """Download and install the pre-trained net.pth file."""
        
        if self.fully_installed:
            print('Data is already fully installed')
            return

        if not os.path.isdir(data_path):
            os.mkdir(data_path)

        print('Starting download, DO NOT STOP UNTIL FINISHED')

        for path in self.paths:
            url = self.paths[path]

            if os.path.isfile(path):
                continue

            print('Downloading', url)

            with open(path, 'wb') as f:
                r = requests.get(url, allow_redirects=True, stream=True)
                total_length = round(float(r.headers['Content-Length']) / 1024 / 1024, 1)
                dl = 0
                start = time.time()
                for chunk in r.iter_content(1024):
                    dl += len(chunk)
                    f.write(chunk)
                    diff = (time.time() - start)
                    if diff == 0: diff = 1
                    sys.stdout.write('\r{}/{}MB, {} MB/s'.format(
                        round(dl / 1024 / 1024, 1),
                        total_length,
                        round((dl / diff) / 1024 / 1024, 1)
                    ))
                print('')

