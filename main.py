import torch
import torchvision
import time
import os
import copy
import random
import pickle
import csv

import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, f1_score
import torch.optim.lr_scheduler as lr

from visualizationFunctions import visualization

def conv_block(in_channels, out_channels, kernel_size, padding, pooling):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.MaxPool3d(pooling),
        nn.ReLU()   # Swap max-pooling and ReLU for more efficiency
    )

class ResConv(nn.Module):
    def __init__(self, features, kernel_size, padding):
        super().__init__()
        self.cnn = nn.Conv3d(features, features, kernel_size=kernel_size, padding=padding)
        self.activation = nn.ReLu()

    def forward(self,inputs):
        return self.activation(self.cnn(inputs))+inputs
        # return self.activation(self.cnn(inputs)+inputs)

class MicroExpressionRecognition3D(nn.Module):
    def __init__(self, dropout_rate=0.1, frames = 32):
        super().__init__()
        self.frames = frames
        self.conv1 = nn.Conv3d(3, 8, kernel_size=(3,3,3), padding=(1,1,1))
        self.activation1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d((1,2,2))
        self.dp1 = nn.Dropout3d(dropout_rate)
        self.conv2 = nn.Conv3d(8, 8, kernel_size=(3,3,3), padding=(1,1,1))
        self.activation2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.dp2 = nn.Dropout3d(dropout_rate)
        self.conv3 = nn.Conv3d(8, 16, kernel_size=(3,3,3), padding=(1,1,1))
        self.activation3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d((1,2,2))
        self.dp3 = nn.Dropout3d(dropout_rate)
        self.conv4 = nn.Conv3d(16, 16, kernel_size=(3,3,3), padding=(1,1,1))
        self.activation4 = nn.ReLU()
        self.pool4 = nn.MaxPool3d((1,2,2))
        self.dp4 = nn.Dropout3d(dropout_rate)
        self.conv5 = nn.Conv3d(16, 32, kernel_size=(3,3,3), padding=(1,1,1))
        self.activation5 = nn.ReLU()
        self.pool5 = nn.MaxPool3d((1,2,2))
        self.dp5 = nn.Dropout3d(dropout_rate)
        self.conv6 = nn.Conv3d(32, 32, kernel_size=(3,3,3), padding=(1,1,1))
        self.activation6 = nn.ReLU()
        #self.pool6 = nn.MaxPool3d((1,2,2))
        # nn.Dropout3d(dropout_rate),
        # conv_block(16, 16, (3, 3, 3), (1, 1, 1), (1, 7, 7)),
        #nn.Conv3d(64,64,kernel_size=(3,3,3),padding=(1,1,1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.frames * 7 * 7 * 32, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        inputs = self.conv1(inputs) #224 x 224 x 8
        inputs = self.activation1(inputs)
        inputs = self.pool1(inputs)
        inputs = self.dp1(inputs)
        inputs = self.conv2(inputs) + inputs #112 x 112 x 16
        inputs = self.activation2(inputs)
        inputs = self.pool2(inputs)
        inputs = self.dp2(inputs)
        inputs = self.conv3(inputs) #residual 112 x 112 x 16
        inputs = self.activation3(inputs)
        inputs = self.pool3(inputs)
        inputs = self.dp3(inputs)
        inputs = self.conv4(inputs) + inputs # 56 x 56 x 32
        inputs = self.activation4(inputs)
        inputs = self.pool4(inputs)
        inputs = self.dp4(inputs)
        inputs = self.conv5(inputs) # residual 56 x 56 x 32
        inputs = self.activation5(inputs)
        inputs = self.pool5(inputs)
        inputs = self.dp5(inputs)
        inputs = self.conv6(inputs) + inputs
        inputs = self.activation6(inputs)
        #inputs = self.pool6(inputs)
        inputs = self.fc(inputs.view(inputs.size(0), -1))
        return inputs


class MEDataset(Dataset):
    def __init__(self, root_dir, transform=None, csv_file="", phase='', path=None, input_size=224, frames = 32):
        self.micro_expressions = pd.read_csv(csv_file, names=['emotion', 'sample', 'clip'])
        self.root_dir = root_dir
        self.transform = transform
        self.phase = phase
        self.emotions = {'positive': 0, 'negative': 1, 'surprise': 2, 'others': 3}
        self.dataset = {'inputs': [], 'targets': []}

        try:
            with open(path, 'rb') as f:
                self.dataset = pickle.load(f)
        except FileNotFoundError:
            for _, row in tqdm(self.micro_expressions.iterrows(), desc='Loading {} dataset'.format(phase)):
                sample = []
                path2 = os.path.join(self.root_dir, self.phase, str(row['emotion']), str(row['sample']), str(row['clip']))
                files = sorted(os.listdir(path2))
                for file in files:
                    image = Image.open(os.path.join(path2, file))
                    image = image.resize((input_size, input_size))
                    sample.append(np.asarray(image))
                    image.close()

                sample = torch.tensor(sample)
                # Transforming using CUDA to speed up, return to CPU to save memory,
                # pending to check memory transfer overhead.
                sample = torch.unsqueeze(sample.permute(3, 0, 1, 2), 0).float().cuda()

                sample = F.interpolate(sample, (frames, input_size, input_size), mode='trilinear', align_corners=False).squeeze()
                self.dataset['inputs'].append((sample).type(torch.ByteTensor))
                self.dataset['targets'].append(torch.tensor(self.emotions[row['emotion']]).long())
                if phase == 'train':
                    transform = transforms.Compose([FlipLR()])
                    sample2 = transform(sample)
                    self.dataset['inputs'].append((sample2).type(torch.ByteTensor))
                    self.dataset['targets'].append(torch.tensor(self.emotions[row['emotion']]).long())
            with open(path, 'wb') as f:
                pickle.dump(self.dataset, f)

        counts = self.micro_expressions['emotion'].value_counts()
        counts = counts['positive'] * [0] + counts['negative'] * [1] + counts['surprise'] * [2] + counts['others'] * [3]
        self.weights = compute_class_weight(class_weight='balanced', classes=np.unique(counts), y=counts)
        self.weights = self.weights / self.weights.sum()
        print(self.weights)

    def __len__(self):
        return len(self.dataset['targets'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'images': self.dataset['inputs'][idx],
            'emotion': self.dataset['targets'][idx]
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    def __call__(self, sample):
        images, emotion = sample['images'], sample['emotion']
        images, emotion = torch.tensor(images), torch.tensor(emotion)
        images = images.permute(3, 0, 1, 2)
        return {'images': images, 'emotion': emotion}


class Normalize(object):
    def __init__(self, norm_value):
        self.norm_value = norm_value

    def __call__(self, sample):
        images, emotion = sample['images'], sample['emotion']
        images, emotion = (images / self.norm_value).float(), emotion.long()
        return {'images': images, 'emotion': emotion}

class MeanNormalize(object):
    def __call__(self, sample):
        images, emotion = sample['images'], sample['emotion']
        images = images.permute(1,0,2,3)
        # mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        for i in range(len(images)):
            mean, std = images[0].mean([1, 2]), images[0].std([1, 2])
            images[i][0] = images[i][0].sub_(mean[0]).div_(std[0])
            images[i][1] = images[i][1].sub_(mean[1]).div_(std[1])
            images[i][2] = images[i][2].sub_(mean[2]).div_(std[2])
        images = images.permute(1, 0, 2, 3)
        return {'images': images, 'emotion': emotion}

class Crop(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if self.p < random.random():
            height = len(sample['images'][2])
            ratio = random.uniform(0.1, 0.2)
            x = random.randint(0, height)
            y = random.randint(0, height)
            shiftX = random.uniform(0, ratio)
            shiftY = ratio - shiftX
            shiftX = int(shiftX * height)
            shiftY = int(shiftY * height)
            if x + shiftX > height:
                temp = x
                x = x - shiftX
                shiftX = temp
            else:
                shiftX = x + shiftX
            if y + shiftY > height:
                temp = y
                y = y - shiftY
                shiftY = temp
            else:
                shiftY = y + shiftY
            image = sample['images']
            for k in range(len(sample['images'][1])):
                for i in range(x-1, shiftX):
                    for j in range(y-1, shiftY):
                        image[0][k][i][j] = 0
                        image[1][k][i][j] = 0
                        image[2][k][i][j] = 0
            sample['images'] = image
        return sample

class FlipLR(object):
    def __call__(self, sample):
        image = sample
        image = torch.flip(image, [3])
        sample = image
        return sample

class FlipLR_random(object):
    def __call__(self, sample, p = 0.5):
        if p < random.random():
            image = sample['images']
            image = torch.flip(image, [3])
            sample['images'] = image
        return sample

class ToFloat(object):
    def __call__(self, sample):
        images, emotion = sample['images'], sample['emotion']
        images, emotion = (images / 255).float(), emotion.long()
        return {'images': images, 'emotion': emotion}

class MakeGray(object):
    def __init__(self, p=0.5):
        self.p = p
        self.to_gray = transforms.Grayscale(num_output_channels=3)

    def __call__(self, sample):
        if self.p < random.random():
            image = sample['images']
            image = image.permute(1, 0, 2, 3)
            image = self.to_gray(image)
            image = image.permute(1, 0, 2, 3)
            sample['images'] = image

        return sample

class MakeJitter(object):
    def __init__(self, p=0.5):
        self.p = p
        self.jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3)

    def __call__(self, sample):
        if self.p < random.random():
            image = sample['images']
            image = image.permute(1, 0, 2, 3)
            image = self.jitter(image)
            image = image.permute(1, 0, 2, 3)
            sample['images'] = image

        return sample



def set_seed(seed=1):
    """
    Sets all random seeds.
    :param seed: int
        Seed value.
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, file_name = "Results.csv", scheduler = None):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    myFile= open(file_name, 'w')
    writer = csv.DictWriter(myFile, fieldnames=('Epoch', 'Train_Loss', 'Train_Acc', 'Train_F1', 'Val_Loss', 'Val_Acc', 'Val_F1'), lineterminator='\n')
    writer.writerow({'Epoch': 'Epoch', 'Train_Loss': 'Train_Loss', 'Train_Acc': 'Train_Acc', 'Train_F1': 'Train_F1',
                     'Val_Loss': 'Val_Loss', 'Val_Acc': 'Val_Acc', 'Val_F1': 'Val_F1'})
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_loss = 0
        train_acc = 0
        train_f1 = 0
        val_loss = 0
        val_acc = 0
        val_f1 = 0
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            val_targets, val_preds = [], []
            running_loss, running_corrects = 0.0, 0
            for inputs in dataloaders[phase]:
                inputs['images'] = inputs['images'].to(device)
                inputs['emotion'] = inputs['emotion'].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs['images'])
                    loss = criterion(outputs, inputs['emotion'])

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    targets_list = inputs['emotion'].cpu().tolist()
                    preds_list = preds.cpu().tolist()
                    val_targets.extend(targets_list)
                    val_preds.extend(preds_list)

                running_loss += loss.item() * inputs['images'].size(0)
                running_corrects += torch.sum(preds == inputs['emotion'])

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)


            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
               best_acc = epoch_acc
               best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                val_acc_history.append(epoch_acc)
                print('Validation Loss: {:.4e} Acc: {:.4f} - Current best Acc: {:.4f}'.format(epoch_loss, epoch_acc, best_acc))
                val_loss = '{:.4f}'.format(epoch_loss)
                val_acc = '{:.4f}'.format(epoch_acc)
            else:
                print('Training Loss: {:.4e} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
                train_loss = '{:.4f}'.format(epoch_loss)
                train_acc = '{:.4f}'.format(epoch_acc)

            conf_matrix = confusion_matrix(val_targets, val_preds)
            f1 = f1_score(y_true=val_targets, y_pred=val_preds, average='weighted')
            if phase == 'val':
                val_f1 = '{:.4f}'.format(f1)
            else:
                train_f1 = '{:.4f}'.format(f1)
            print(conf_matrix, f1)
        writer.writerow({'Epoch': epoch, 'Train_Loss': train_loss, 'Train_Acc': train_acc, 'Train_F1': train_f1,
                         'Val_Loss': val_loss,'Val_Acc': val_acc, 'Val_F1': val_f1})
        # scheduler.step()
        # print(optimizer)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.eval()
    test_targets, test_preds = [], []
    running_loss, running_corrects = 0.0, 0
    for inputs in dataloaders['test']:

        inputs['images'] = inputs['images'].to(device)
        inputs['emotion'] = inputs['emotion'].to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled('test' == 'train'):
            outputs = model(inputs['images'])
            loss = criterion(outputs, inputs['emotion'])
            _, preds = torch.max(outputs, 1)
            targets_list = inputs['emotion'].cpu().tolist()
            preds_list = preds.cpu().tolist()
            test_targets.extend(targets_list)
            test_preds.extend(preds_list)

        running_loss += loss.item() * inputs['images'].size(0)
        running_corrects += torch.sum(preds == inputs['emotion'])

    test_loss = running_loss / len(dataloaders['test'].dataset)
    test_acc = running_corrects.double() / len(dataloaders['test'].dataset)
    print('Test Loss: {:.4e} Acc: {:.4f}'.format(test_loss, test_acc))
    test_loss = '{:.4f}'.format(test_loss)
    test_acc = '{:.4f}'.format(test_acc)
    conf_matrix = confusion_matrix(test_targets, test_preds)
    f1 = f1_score(y_true=test_targets, y_pred=test_preds, average='weighted')
    test_f1 = '{:.4f}'.format(f1)
    myFile.close()
    print(conf_matrix, f1)
    with open(file_name, 'a', newline='') as f:
        writer1 = csv.writer(f)
        writer1.writerow([""])
        writer1.writerow(["Test Loss", test_loss, ""])
        writer1.writerow(["Test Acc", test_acc, ""])
        writer1.writerow(["Test F1", test_f1, ""])
        writer1.writerows(conf_matrix)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__ == '__main__':
    data_dir = os.path.join('G:\\tesina\\Licencias', 'MicroExpressions_Data2')
    learning_rate = 1e-4
    weight_decay = 1e-2
    dropout_rate = 0.1
    num_classes = 4
    batch_size = 8
    num_epochs = 150
    input_size = 224
    num_workers = 10
    frames = 32
    cross_val = 19
    file_name = "G:\\tesina\\Licencias\\Results\\" + "STCNN" + str(cross_val) + "_LR" + str(learning_rate) + "_WD" + str(weight_decay) + "_DR" + str(dropout_rate) + "_BS" + str(batch_size) + "_F" + str(frames) + ".csv"

    train_transforms = transforms.Compose([
        ToFloat(),
        transforms.RandomChoice([MakeGray(), MakeJitter()], p=[0.5, 0.5]),
        Crop(),
        MeanNormalize()
    ])

    val_transforms = transforms.Compose([
        ToFloat(),
        MeanNormalize()
    ])

    print("Initializing Datasets and Dataloaders...")
    image_datasets = {
        'train': MEDataset(root_dir=os.path.join('G:\\tesina\\Licencias', 'MicroExpressions_Data2'), transform=train_transforms,
                           csv_file='train_data.csv', phase='train', path='train.pkl', input_size=input_size, frames = frames),
        'val': MEDataset(root_dir=os.path.join('G:\\tesina\\Licencias', 'MicroExpressions_Data2'), transform=val_transforms,
                         csv_file='val_data.csv', phase='val', path='val.pkl', input_size=input_size, frames = frames),
        'test': MEDataset(root_dir=os.path.join('G:\\tesina\\Licencias', 'MicroExpressions_Data2'), transform=val_transforms,
                         csv_file='test_data.csv', phase='test', path='test.pkl', input_size=input_size, frames=frames)
    }

    data_loaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    torch.cuda.empty_cache()
    set_seed(10)
    model_ft = MicroExpressionRecognition3D(dropout_rate, frames)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    print(device)
    #optimizer = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    optimizer = optim.Adam(model_ft.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #scheduler = lr.StepLR(optimizer, step_size=20, gamma=0.01)
    weights = torch.from_numpy(image_datasets['train'].weights).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    model_ft, hist = train_model(model_ft, data_loaders, criterion, optimizer, num_epochs=num_epochs,
                                 file_name=file_name)
    with open(file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([""])
        writer.writerow(["Weight Decay", weight_decay, ""])
        writer.writerow(["Learning Rate", learning_rate, ""])
        writer.writerow(["Frames", frames, ""])
        writer.writerow(["Batch Size", batch_size, ""])
        writer.writerow(["Input Size", input_size, ""])
        writer.writerow(["Dropout Rate", dropout_rate, ""])

    # visualization(model_ft, 'activation6', 'G:\\tesina\\Licencias\\Prueba_happiness', 'G:\\tesina\Figuras\Heatmaps\\', 'G:\\tesina\Figuras\Gifs\\', input_size)