## Utils

from datasets import load_dataset, Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ColorJitter, ToTensor, Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomAffine
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from time import time
import torch.utils.tensorboard as tb
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torchvision.transforms
from torch import save
from os import path

class MyDataset(Dataset):
    def __init__(self, data_path, train=True):

        if train:
            self.dataset = load_dataset("imagefolder", data_dir=data_path, drop_labels=False, split="train")
            self.dataset = self.dataset.map(self.img_resize, remove_columns=["image"], batched=True)
            self.dataset.set_transform(self.transforms)
        else:
            self.dataset = load_dataset("imagefolder", data_dir=data_path, drop_labels=False, split="test")
            self.dataset = self.dataset.map(self.img_resize, remove_columns=["image"], batched=True)
            self.dataset.set_transform(self.test_transform)

    def transforms(self, imgs):
        augment = Compose([
                            RandomHorizontalFlip(p=0.5), 
                            RandomVerticalFlip(p=0.5),
                            ColorJitter(brightness=0.1,
                                        contrast=0.1,
                                        saturation=0.1,
                                        hue=0),
                                RandomRotation(degrees=45),
                                RandomAffine(degrees=10),
                                ToTensor()
                            ])
        imgs["pixel_values"] = [augment(image) for image in imgs["pixel_values"]]
        return imgs

    def test_transform(self, imgs):
        augment = Compose([ToTensor()])
        imgs["pixel_values"] = [augment(image) for image in imgs["pixel_values"]]
        return imgs

    def img_resize(self, imgs):
        imgs["pixel_values"] = [image.convert("RGB").resize((100,100)) for image in imgs["image"]]
        return imgs

    def __getitem__(self, index):
        data = self.dataset[index]
        label = F.one_hot(torch.tensor(data["label"]), num_classes=3)
        return data["pixel_values"], label.float()

    def __len__(self):
        return len(self.dataset)
#bean_data_train = MyDataset("../../beans",train)

#img = bean_data_train[0]["pixel_values"]
#plt.imshow(np.transpose(img, (1,2,0)))


#type(bean_data_train)

## Build the Model


"""
CNN class
 - Can take layers argument to define number of channels and depth
 - Number of input channels will always be 3
 - Currently the first layer is hardcoded with kernel size 7, and stride 2
    I think this should be reduced to 5 or even 3.
 - The class Block defines a block of 2 conv layers. This could be extended to 3
    and include a skip. Could also include params for kernal size and striding
 - Normalization is performed here instead of in Utils 
"""

class ConvoClassifier(torch.nn.Module):

    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()

            # Defines a two layer block with stride in first layer only, batch norm after each
            self.net = torch.nn.Sequential(
                # Only the first layer is strided, can adjust this in the loop in the init method
                torch.nn.Conv2d(n_input, n_output, kernel_size=3,
                                padding=1, stride=stride),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self, layers=[32, 64, 128], n_input_channels=3, n_classes=3):
        super().__init__()
        # Inital layer with kernal size 7, the max pool appears to increase accuracy on validation set
        L = [torch.nn.Conv2d(n_input_channels, layers[0], kernel_size=7, padding=3, stride=2),
             torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        c = layers[0]
        # Build network from list of layers
        for l in layers:
            # can adjust stride here
            L.append(self.Block(c, l, stride=2))
            c = l

        self.network = torch.nn.Sequential(*L)
        # Linear layer at end for the 3 classification labels
        self.classifier = torch.nn.Linear(c, n_classes)
        # Mean and standard dev of color channels accross the entire training set
        self.norm = torchvision.transforms.Normalize(
            mean=[0.233, 0.298, 0.256], std=[0.199, 0.118, 0.201])

    def forward(self, x):
        # Normalize
        normx = self.norm(x)
        # Compute the features
        z = self.network(normx)
        # Global average pooling
        z = z.mean(dim=[2, 3])
        # Classify
        return self.classifier(z)

# Save the model with epoch number and message/name of model (for checkpoints)
def save_model(model, message, epoch):
    name = message + '_' + str(epoch) + '_' + 'det.th'
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), name))

def load_model(model_name):
    from torch import load
    from os import path
    r = ConvoClassifier()
    r.load_state_dict(load(path.join(path.dirname(
        path.abspath(__file__)), model_name), map_location='cpu'))
    return r




## Train the Model

#from .models import ConvoClassifier, save_model, load_model
#from .utils import MyDataset

"""
Running tensorboard
 - launch terminal w/deeplearning virual env
 - run python -m tensorboard.main --logdir=runs
 - open in browser
 - enabels visualization of training loss and accuracy after each batch
    and validation accuracy after each epoch
"""

"""
Main training loop
 - Takes training arguments
    - log_dir: directory of logs for tensorboard
    - run_info: short description of run for identification in tensorboard
    - lr: learning rate
    - ep: number of epochs
    - layers: takes multiple int values and constructs a list used for construction of model. 
        Each number is number of channels and length of list is number of layers

 - Prints time and validation accuracy to consol after each epoch. Saves model at end
 - Note: each "layer" is a block of 2 convolutional layers, see models.py
 - Should add ability to customize learning rate schedule, currently decaying around 
    6 epochs gives good results 
"""

def load_data(dataset_path, num_workers=0, batch_size=256, train=True):
    dataset = MyDataset(dataset_path, train)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def train(args):

    start = time()
    # model constructed here
    model = ConvoClassifier(args.layer_list, args.num_classes).to(device)

    # set up logger with the run info name
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_dir = "train" + args.run_info
        valid_dir = "valid" + args.run_info
        train_logger = tb.SummaryWriter(path.join(args.log_dir, train_dir))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, valid_dir))

    # Choice of optimizer, adam working better so far
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

    # LR scheduler, will want to eventually add ability to customize args for this
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=6, gamma=0.2)
    step = 0


    train_data = load_data(args.data_dir,0,args.batch_size,True)
    val_data = load_data(args.data_dir,0,args.batch_size,False)

    # Main loop
    for epoch in range(args.num_epochs):
        startepoch = time()
        total_loss = 0

        # Make sure things are set to training mode
        model.train()
        for i, (x, y) in enumerate(train_data):

            x = x.to(device)
            y = y.to(device)
            output = model(x)
            l = F.cross_entropy(output, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            total_loss += l

            # compute accuracy on training batch, need to get it back to CPU and change to numpy array
            acc = (output.argmax(1).type_as(y) ==
                   y.argmax(1)).float().detach().cpu().numpy()
            acc = np.mean(acc)

            train_logger.add_scalar("Loss", l, global_step=step)
            #train_logger.add_scalar("acc", acc, global_step=step)
            step += 1

        # Test model on validation set after training epoch, make sure to set to eval mode
        model.eval()
        val_acc = np.array([])
        for i, (x, y) in enumerate(val_data):
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            # compute accuracy on validation set, need to get it back to CPU and change to numpy array
            acc = (output.argmax(1).type_as(y) ==
                   y.argmax(1)).float().detach().cpu().numpy()
            acc = np.mean(acc)
            val_acc = np.append(val_acc, acc)
           

        valid_logger.add_scalar(
            "val_acc_epoch", np.mean(val_acc), global_step=step)
       
        # End of epoch, print validation accurcy and epoch time
        endepoch = time()
        scheduler.step()
        print(np.mean(val_acc))
        print("epochtime", endepoch-startepoch)

    # print total time of model and save
    end = time()
    print("total time", end-start)
    save_model(model, args.run_info, args.num_epochs)


"""
Arguments:
 - log_dir: directory of logs for tensorboard
 - run_info: short description of run for identification in tensorboard
 - lr: learning rate
 - ep: number of epochs
 - layers: takes multiple int values and constructs a list used for construction of model. 
    Each number is number of channels and length of list is number of layers
"""

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default='runs')
    parser.add_argument('-n', '--run_info', type=str)
    parser.add_argument('-lr', '--learn_rate', type=float, default=0.0001)
    parser.add_argument('-ep', '--num_epochs', type=int, default=4)
    # layer list requires at least one number. Multiple numbers seperated by a single space
    parser.add_argument('-layers', '--layer_list', nargs='+',
                        type=int, default=[32, 64, 128])
    parser.add_argument('-data', '--data_dir', type=str, default='../../beans')
    parser.add_argument('-c', '--num_classes', type=int, default=3 )
    parser.add_argument('-bs', '--batch_size', type=int, default=256)

    args = parser.parse_args()
    train(args)
