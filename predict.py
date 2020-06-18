import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.image as mpimg
import random
import os
import torch.utils.data as data
from torchvision import datasets
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from os.path import basename
from torchvision.models import mobilenet_v2


# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Reading input folder
#files = os.listdir(args.input_folder)

#####
# TODO - your prediction code here
# Hyper Parameters
num_epochs = 10
batch_size = 64
learning_rate = 0.001

class ImageFolderWithName(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.images_paths = glob(root + '/*.jpg')
        self.labels = torch.LongTensor([int(basename(image_path).split('_')[1].split('.')[0]) for image_path in self.images_paths])

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        image = self.transform(Image.open(image_path))
        label = self.labels[idx]
        return (image, label, basename(image_path))


transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

image_datasets = ImageFolderWithName(args.input_folder, transform=transform_test)

data_loader = data.DataLoader(image_datasets,
                                batch_size=batch_size,
                                shuffle=False)
def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x
def evaluation_acc(labels, outputs):
    return accuracy_score(labels, outputs)

#Define the Model
Net = mobilenet_v2()
Net.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=False),nn.Linear(in_features=1280, out_features=2, bias=True))
Net.load_state_dict(torch.load("net_model_1.pt"))
Net = to_gpu(Net)

criterion = to_gpu(nn.CrossEntropyLoss())
optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)

labels_arr = []
predicted_arr = []
names_arr = []
Net.eval()
for images, labels, names in data_loader:
    images = to_gpu(images)
    labels = to_gpu(labels)
    outputs = Net(images)
    foo, predicted = torch.max(outputs.data, 1)
    predicted_arr = np.append(predicted_arr, predicted.cpu().numpy())
    labels_arr = np.append(labels_arr, labels.cpu().numpy())
    names_arr = np.append(names_arr, list(names))
    loss = criterion(outputs, labels)

print("Accuracy: ", evaluation_acc(labels_arr, predicted_arr))

# Example (A VERY BAD ONE):
y_pred = predicted_arr
prediction_df = pd.DataFrame(zip(names_arr, y_pred), columns=['id', 'label'])
####

# TODO - How to export prediction results
prediction_df.to_csv("prediction.csv", index=False, header=False)


# ### Example - Calculating F1 Score using sklrean.metrics.f1_score
y_true = labels_arr
f1 = f1_score(y_true, y_pred, average='binary')		# Averaging as 'binary' - This is how we will evaluate your results.

print("F1 Score is: {:.2f}".format(f1))


