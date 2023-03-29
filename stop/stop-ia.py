__author__ = "Amaury COLIN"
__credits__ = "Amaury COLIN"
__date__ = "2023.12.23"
__version__ = "1.1.0"

import math
import time
import os
from tkinter import N
import cv2
import numpy as np
import copy
import json
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from alive_progress import alive_bar 

class StopLineDataset(Dataset):
    """Stop Lines dataset."""

    def __init__(self, labels_file, root_dir, transform=None):
        """
        Args:
            labels_file (string): Path to the json file with labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        labels = open(labels_file, 'r')
        labels = json.loads(labels.read())
        self.labels = labels
        
        self.root_dir = root_dir
        
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        transformImg=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()]) 
        
        img_name = os.path.join(self.root_dir,
                                self.labels[idx]['img_name'])
        
        # plt.imshow(cv2.imread(img_name))
        # plt.show()
        # print(img_name)
        image = transformImg(cv2.imread(img_name))
        label_value = torch.tensor([float(self.labels[idx]['label_value'])])
        sample = {'image': image, 'label_value': label_value, 'idx':  torch.tensor([float(idx)])}

        # if self.transform:
        #     sample = self.transform(sample)

        return sample


class Classif:
    
    datasets = {}
    
    def __init__(self, device = False):
        
        self.classes = []
        self.decision = Decision(0.3)
        
        self.epoch_losses = []
        self.epoch_accs = []
        self.errors_last_epoch = []
        self.preds_each_epochs = []
        
        if not(device):
            self.set_device()
        else:
            self.device = device
            
    def load_model(self, model):
        if isinstance(model, str):
            self.model = torch.load(model)
        else:
            self.model = model
             
        self.model.to(self.device)

    def set_device(self):
        torch.cuda.is_available()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print(device)
        
    def get_device(self):
        return self.device

    def set_dataset(self, dataset_name: str, dataset_dir: str, data_transform = transforms.Compose([transforms.ToTensor()]), data_batch_size = 4):
            
        SL_dataset = StopLineDataset(dataset_dir + "/labels.json", dataset_dir, data_transform)
        
        dataload = DataLoader(SL_dataset, batch_size=data_batch_size, shuffle=True, num_workers=0)
        
        self.datasets[dataset_name] = {
            'dir': dataset_dir,
            'images': SL_dataset,
            'dataloader': dataload,
            'batch_size': data_batch_size,
            'size': len(SL_dataset),
            'labels_file': dataset_dir + "/labels.json",
            'labels': json.loads((open(dataset_dir + "/labels.json", "r")).read())
        }
    
    def set_classes(self, classes):
        self.classes = classes
    
    def get_dataset(self, dataset_name: str):
        if dataset_name in self.datasets.keys():
            return self.datasets[dataset_name]
        else:
            raise Exception("Le dataset n'a pas été trouvé. A-t-il été créé ?")
    
    def set_criterion(self, criterion):
        self.criterion = criterion
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
    
    def train(self, train_dataset_name, eval_dataset_name, num_epochs=5):
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        
        dataloader_train = self.get_dataset(train_dataset_name)['dataloader']
        dataloader_eval = self.get_dataset(eval_dataset_name)['dataloader']
        
        bar_total = num_epochs *(len(dataloader_train) + len(dataloader_eval))
        with alive_bar(bar_total) as bar:
        
            for epoch in range(num_epochs):
                print(f'Epoch {epoch + 1}/{num_epochs}')
                print('-' * 10)
                
                errors = {
                    'train': [],
                    'eval': [],
                }

                n = {
                    train_dataset_name: 0,
                    eval_dataset_name: 0
                }
                
                self.preds_each_epochs.append([])

                for phase in [train_dataset_name, eval_dataset_name]:

                # Each epoch has a training and validation phase
                    if phase == train_dataset_name:
                        self.model.train()  # Set model to training mode
                    else:
                        self.model.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0
                    print("Phase en cours : ", phase)
                    print()

                    dataloader = self.get_dataset(phase)['dataloader']
                    
                    
                    preds_batch_i = [[None, None] for k in range(len(dataloader))]

                    for i_batch, sample_batched in enumerate(dataloader):
                        inputs = sample_batched['image'].to(self.device)
                        labels = sample_batched['label_value'].to(self.device)
                        image_ids = sample_batched['idx'].to(self.device)
                                
                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == train_dataset_name):
                            outputs = self.model(inputs)
                            preds_batch_i[i_batch][0] = list(outputs)
                            preds_batch_i[i_batch][1] = list(labels)
                            
                            loss = self.criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == train_dataset_name:
                                loss.backward()
                                self.optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        print("Loss : ", loss.item())
                        for c in range(len(preds_batch_i[i_batch][0])):
                            n[phase] += 1
                            
                            if phase == train_dataset_name:
                                if float(self.decision.binary(preds_batch_i[i_batch][0][c])) != float(preds_batch_i[i_batch][1][c]):
                                    errors['train'].append(self.get_dataset(train_dataset_name)['labels'][int(image_ids[c])])
                                else:
                                    running_corrects += 1
                            elif phase == eval_dataset_name:
                                
                                self.preds_each_epochs[len(self.preds_each_epochs) - 1].append([preds_batch_i[i_batch][0][c], preds_batch_i[i_batch][1][c]])
                                
                                if float(self.decision.binary(preds_batch_i[i_batch][0][c])) != float(preds_batch_i[i_batch][1][c]):
                                    errors['eval'].append(self.get_dataset(eval_dataset_name)['labels'][int(image_ids[c])])
                                else:
                                    running_corrects += 1
                        
                        print("Erreurs : ", len(errors['train']) / n[train_dataset_name], 0 if n[eval_dataset_name] == 0 else (len(errors['eval']) / n[eval_dataset_name]))
                        bar()
                        
                    print()
                    print("Résultat de la phase (" + phase + ") : ")
                    
                    if phase == train_dataset_name:
                        self.scheduler.step()

                    epoch_loss = running_loss / self.get_dataset(phase)['size']
                    epoch_acc = running_corrects / self.get_dataset(phase)['size']
                    
                    self.epoch_losses.append(epoch_loss)
                    self.epoch_accs.append(epoch_acc)
                    
                    self.errors_last_epoch = errors

                    print(f'{phase} - Epoch loss : {epoch_loss:.4f} - Epoch acc : {epoch_acc:.4f}')

                    # deep copy the model
                    if phase == eval_dataset_name:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.model.state_dict())

                print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val test: {best_acc:4f}')

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model

    def validate(self, validation_dataset_name):
        successes = {
            'is_0': [],
            'is_1': []
        }
        errors = {
            'should_be_0': [],
            'should_be_1': [],
        }
        
        dataloader = self.get_dataset(validation_dataset_name)['dataloader']
        preds_batch_i = [[None, None] for k in range(len(dataloader))]
        
        for i_batch, sample_batched in enumerate(dataloader):
                # get the inputs
            inputs = sample_batched['image'].to(self.device)
            labels = sample_batched['label_value'].to(self.device)
            image_ids = sample_batched['idx'].to(self.device)
                
            outputs = self.model(inputs)
            preds_batch_i[i_batch][0] = list(outputs)
            preds_batch_i[i_batch][1] = list(labels)
            
            for c in range(len(preds_batch_i[i_batch][0])):
                if float(self.decision.binary(preds_batch_i[i_batch][0][c])) != float(preds_batch_i[i_batch][1][c]):
                    if preds_batch_i[i_batch][1][c] == 1:
                        errors['should_be_1'].append((self.get_dataset(validation_dataset_name)['labels'][int(image_ids[c])], float(preds_batch_i[i_batch][0][c])))
                    else:
                        errors['should_be_0'].append((self.get_dataset(validation_dataset_name)['labels'][int(image_ids[c])], float(preds_batch_i[i_batch][0][c])))
                else:
                    if preds_batch_i[i_batch][1][c] == 1:
                        successes['is_1'].append(self.get_dataset(validation_dataset_name)['labels'][int(image_ids[c])])
                    else:
                        successes['is_0'].append(self.get_dataset(validation_dataset_name)['labels'][int(image_ids[c])])
            
        nb = {
            'total': len(errors['should_be_0']) + len(errors['should_be_1']) + len(successes['is_0']) + len(successes['is_1']),
            'errors': {
                'should_be_0': len(errors['should_be_0']),
                'should_be_1': len(errors['should_be_1']),
                'total': len(errors['should_be_0']) + len(errors['should_be_1']),
            },
            'successes': {
                'is_0': len(successes['is_0']),
                'is_1': len(successes['is_1']),
                'total': len(successes['is_0']) + len(successes['is_1']),
            }
        }
        ratio = {
            'errors': {
                'should_be_0': nb['errors']['should_be_0'] / nb['errors']['total'],
                'should_be_1': nb['errors']['should_be_1'] / nb['errors']['total'],
            },
            'successes': {
                'is_0': nb['successes']['is_0'] / nb['successes']['total'],
                'is_1': nb['successes']['is_1'] / nb['successes']['total']
            }
        }
        return ratio, nb, successes, errors
    
    def show_stats(self):
        fig,ax = plt.subplots()
        
        ax.plot([self.epoch_losses[i] for i in range(len(self.epoch_losses)) if i % 2 == 1], color="red", label="Loss")
        ax.set_xlabel("Loss")
        
        ax2=ax.twinx()
        ax2.plot(self.epoch_accs, color="green", label="Acc")
        ax2.set_xlabel("Accuracy")
        plt.show()
        
        e = self.errors_last_epoch['eval'][:10]
        print(e)
        fig = plt.figure(figsize=(10, 7))
        rows = 2
        columns = math.ceil(len(e) / 2)
        
        for c in range(len(e)):
            fig.add_subplot(rows, columns, c + 1)
            plt.imshow(cv2.imread(self.get_dataset('TEST')['dir']  + e[c]['img_name']))
            plt.axis('off')
            plt.title(e[c]['label_value'])
            
        plt.show()

    def test_several_thresholds(self, thresholds = [0.95]):
        fig,ax = plt.subplots()
        
        l, = ax.plot([self.epoch_losses[i] for i in range(len(self.epoch_losses)) if i % 2 == 1], color="red", label="Loss")
        ax.set_xlabel("Loss")
        l.set_label("Loss")
        ax.legend()
        
        ax2=ax.twinx()
        ax2.set_xlabel("Accuracy")
        print(thresholds)
        for threshold in thresholds:
            print(threshold)
            decidator = Decision(float(threshold))
            epoch_accs = []
            for preds_epoch in self.preds_each_epochs:
                running_corrects = 0
                for img in range(len(preds_epoch)):
                    if float(decidator.binary(preds_epoch[img][0])) == float(preds_epoch[img][1]):
                        running_corrects += 1
                epoch_accs.append(running_corrects / len(preds_epoch))
            print(epoch_accs)
            e, = ax2.plot(epoch_accs)
            e.set_label('Thr' + str(threshold))
        ax2.legend() 
        plt.show()
                
    
    def test_image(self, image_path):
        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_image = image.copy()
        
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        
        with torch.no_grad():
            outputs = self.model(image.to(self.device))
            print(float(outputs[0]), outputs)
            output_label = 1 if THRESHOLD < float(outputs[0]) else 0
            
        return (orig_image, output_label)
        
    def get(self):
        return self.model

    def save(self, save_filename):
        torch.save(self.model, save_filename)



class Decision():
    def __init__(self, threshold) -> None:
        self.threshold = threshold
    
    def binary(self, value):
        if value > self.threshold:
            return 1
        return 0

if __name__ == "__main__":
    
    ROOT_DIR = "./stop/"
    
    MODELE_ENREGISTRE = False
    ENREGISTRER_MODELE = True
    MODELE_NAME = "sigmoid_sigma_crop40.pth"
    
    N_EPOCHS = 15
    BATCH_SIZE_TRAIN = 10
    BATCH_SIZE_VALIDATE = 10
    
    THRESHOLD = 0.9
    
    if not(MODELE_ENREGISTRE):
        m = models.resnet18(pretrained=True)
        m.fc = nn.Sequential(
            nn.Linear(m.fc.in_features, 1),
            nn.Sigmoid()
        )
    else:
        m = ROOT_DIR + MODELE_NAME
    
    c = Classif()
    c.load_model(m)
    
    if not(MODELE_ENREGISTRE):
        c.set_criterion(nn.BCELoss())
        c.set_optimizer(optim.SGD(c.model.parameters(), lr=0.001, momentum=0.9))
        c.scheduler = lr_scheduler.StepLR(c.optimizer, step_size=7, gamma=0.1)

        c.set_dataset('TRAIN', ROOT_DIR + 'dataset_sigma_crop/train', data_batch_size=BATCH_SIZE_TRAIN)
        c.set_dataset('TEST', ROOT_DIR + 'dataset_sigma_crop/test', data_batch_size=BATCH_SIZE_TRAIN)
    
    c.set_dataset('VALIDATION', ROOT_DIR + 'validation_dataset_sigma_crop', data_batch_size=BATCH_SIZE_VALIDATE)
    # c.set_classes(['sans', 'avec'])
    
    if not(MODELE_ENREGISTRE):
        c.train('TRAIN', 'TEST', N_EPOCHS)
        # c.show_stats()
        c.test_several_thresholds([0.1, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99])
    
    if ENREGISTRER_MODELE:
        c.save(ROOT_DIR + MODELE_NAME)
    
    r, n, s, e = c.validate('VALIDATION')
    
    print("Nombre d'images : ", n['total'])
    print()
    print("Détection de l'absence de ligne :")
    print("   Erreurs :", n['errors']['should_be_0'])
    print("   Réussites :", n['successes']['is_0'])
    print()
    print("Détection de la présence de ligne :")
    print("   Erreurs :", n['errors']['should_be_1'])
    print("   Réussites :", n['successes']['is_1'])
    
    print(e)
    
    # test = c.test_image(ROOT_DIR + "validation_dataset/class0/1431_cam_image_array_.jpg")
    # fig = plt.figure(0)
    # fig.suptitle(test[1]) 
    # plt.imshow(test[0])
    # plt.show()
