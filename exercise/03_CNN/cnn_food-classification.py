# ---import modules---#
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import time


# ---read files from directory---#
def readfiles(work_path, label):
    """
    define a function to read images into data set with the help of OpenCV (cv2)
    :param work_path: directory of images
    :param label: boolean variable, which indicates if y should be returned or not
    :return: numpy data
    """
    image_dir = os.listdir(work_path)
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.unit8)  # the range of data type "unit8" is [0,255), it is
    # suitable to handle with image pixel
    y = np.zeros((len(image_dir)), dtype=np.unit8)
    for i, file in enumerate(image_dir):
        image = cv2.imread(os.path.join(work_path, file))
        x[i, :, :] = cv2.resize(image, (128, 128))
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x


# generate training set, validation set and testing set via "readfiles" function
path = "./food-11/food-11"
print("Reading data...")
train_x, train_y = readfiles(os.path.join(path, 'training'), label=True)
print("the size of training data is {}".format(len(train_x)))
val_x, val_y = readfiles(os.path.join(path, 'validation'), label=True)
print("the size of validation data is {}".format(len(val_x)))
test_x = readfiles(os.path.join(path, 'test'), label=False)
print("the size of testing data is {}".format(len(test_x)))

"""
# save these image raw data sets after the first run, later load data sets directly, in order to save time
np.savez("num_data.npz", train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, test_x=test_x)

numpy_data = np.load("num_data.npz")
train_x = numpy_data["train_x"]
train_y = numpy_data["train_y"]
val_x = numpy_data["val_x"]
val_y = numpy_data["val_y"]
test_x = numpy_data["test_x"]
"""


# --- prepare dataset for nn with Dataset and DataLodader ---#
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a long tensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


# generate data set for nn
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    # Horizontally flip the given PIL Image randomly with a given probability, default is 50%
    transforms.RandomRotation(15),  # Rotate the image by angle
    transforms.ToTensor()  # also with data normalization [0.0, 1.0]
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

batch_size = 8  # tune with own GPU capacity
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, train_transform)
train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_data = DataLoader(val_set, batch_size=batch_size, shuffle=False)


# --- neural network model ---#
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # input dimension [3,128,128]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # output [64,128,128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # output [64,64,64]

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # output [128,64,64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # output [128,32,32]

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # output [256,32,32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # output [256,16,16]

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # output [512,16,16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # output [512,8,8]

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # output [512,8,8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # output [512,4,4]
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


# ---Training data---#
num_epoch = 30
learning_rate = 1e-3

model = Model().cuda()
loss = nn.CrossEntropyLoss()  # as it is regarding to multi-classification, loss function is chosen as Crossentropy
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("training model...")
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()
    for i, data in enumerate(train_data):
        optimizer.zero_grad()
        train_pred = model(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())  # prediction and label must be on cpu or gpu at the same time
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[
            1].numpy())  # calculate the number of cases which classes are rightly classified
        # carefully modified for different questions
        train_loss += batch_loss.item()

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_data):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

    # print the result for each epoch
    print("(%03d/%03d) %2.2f sec(s) Train acc:%3.6f Loss:%3.6f | Val acc:%3.6f Loss:%3.6f" % (
        epoch + 1, num_epoch + 1, time.time() - epoch_start_time, train_acc / len(train_x), train_loss / len(train_x),
        val_acc / len(val_x), val_loss / len(val_x)
    ))

# ---Testing---#
test_set = ImgDataset(test_x, transform=test_transform)
test_data = DataLoader(test_set, batch_size=batch_size, shuffle=False)
pred_result = []

print("testing the model...")
model.eval()
with torch.no_grad():
    for i, data in enumerate(test_data):
        test_pred = model(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            pred_result.append(y)

# ---write the prediction result in csv file---#
print("writing the prediction result in prediction.csv file")
with open("prediction.csv", 'w') as f:
    f.write("ID,Category\n")
    for i, y in enumerate(pred_result):
        f.write("{},{}\n".format(i, y))

print("finished!")
