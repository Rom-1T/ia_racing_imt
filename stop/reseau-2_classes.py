import time
import os
import cv2
import numpy as np
import copy
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from alive_progress import alive_bar 

class Classif:
    
    datasets = {}
    
    def __init__(self, device = False):
        
        self.classes = []
        
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
        image_dataset = datasets.ImageFolder(dataset_dir, data_transform)
        dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=data_batch_size, shuffle=True)
        dataset_size = len(image_dataset)
        
        self.datasets[dataset_name] = {
            'images': image_dataset,
            'dataloader': dataloader,
            'batch_size': data_batch_size,
            'size': dataset_size
        }
        
        if len(self.classes) == 0:
            self.set_classes(self.datasets[dataset_name]['images'].classes)    
    
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
        
        bar_total = num_epochs *(len(self.get_dataset(train_dataset_name)['dataloader']) + len(self.get_dataset(eval_dataset_name)['dataloader']))
        with alive_bar(bar_total) as bar:
        
            for epoch in range(num_epochs):
                print(f'Epoch {epoch + 1}/{num_epochs}')
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in [train_dataset_name, eval_dataset_name]:
                    if phase == train_dataset_name:
                        self.model.train()  # Set model to training mode
                    else:
                        self.model.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0
                    print("Phase en cours : ", phase)
                    print()
                    
                    ls = [0 for k in range(len(self.classes))]
                    # label0=0
                    # label1=0
                    # Iterate over data.
                    for inputs, labels in self.get_dataset(phase)['dataloader']:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        for i in labels:
                            ls[i] += 1
                                
                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == train_dataset_name):
                            outputs = self.model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = self.criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == train_dataset_name:
                                loss.backward()
                                self.optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        
                        bar()
                        
                    print()
                    print("Résultat de la phase (" + phase + ") : ")
                    class_names = self.classes
                    for k in range(len(class_names)):
                        print(class_names[k],ls[k])
                    
                    if phase == train_dataset_name:
                        self.scheduler.step()

                    epoch_loss = running_loss / self.get_dataset(phase)['size']
                    epoch_acc = running_corrects.double() / self.get_dataset(phase)['size']

                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                    # deep copy the model
                    if phase == eval_dataset_name and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.model.state_dict())

                print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val test: {best_acc:4f}')

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model

    def validate(self, validation_dataset_name, num_epochs = 5):
        total_count = 0
        classes = self.classes
        successes = [0 for i in range(len(classes))]
        errors = [0 for i in range(len(classes))]
        
        for i, (images, labels) in enumerate(self.get_dataset(validation_dataset_name)['dataloader'], 0):
                # get the inputs
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)   
                
                _, predicted = torch.max(outputs, 1)
                
                for j in range(len(predicted)):
                    total_count += 1
                    if predicted[j] == labels[j]:
                        successes[predicted[j]] += 1
                    else:
                        errors[predicted[j]] += 1
        
        return (total_count, successes, errors) 
        
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
            output_label = torch.topk(outputs, 1)

        print(int(output_label.indices))

        pred_class = self.classes[int(output_label.indices)]
        
        return (orig_image, pred_class)
        
    def get(self):
        return self.model

    def save(self, save_filename):
        torch.save(self.model, save_filename)
        
if __name__ == "__main__":
    
    ROOT_DIR = "./stop/"
    
    MODELE_ENREGISTRE = False
    ENREGISTRER_MODELE = False
    
    N_EPOCHS = 5
    BATCH_SIZE_TRAIN = 5
    BATCH_SIZE_VALIDATE = 10
    
    if not(MODELE_ENREGISTRE):
        m = models.resnet18(pretrained=True)
        m.fc = nn.Linear(m.fc.in_features, 2)
    else:
        m = ROOT_DIR + "rsnt18.pth"
    
    c = Classif()
    c.load_model(m)
    
    if not(MODELE_ENREGISTRE):
        c.set_criterion(nn.CrossEntropyLoss())
        c.set_optimizer(optim.SGD(c.model.parameters(), lr=0.001, momentum=0.9))
        c.scheduler = lr_scheduler.StepLR(c.optimizer, step_size=7, gamma=0.1)

    

        c.set_dataset('TRAIN', ROOT_DIR + 'ultimate_dataset/train', data_batch_size=BATCH_SIZE_TRAIN)
        c.set_dataset('TEST', ROOT_DIR + 'ultimate_dataset/test', data_batch_size=BATCH_SIZE_TRAIN)
    
    c.set_dataset('VALIDATION', ROOT_DIR + 'validation_dataset', data_batch_size=BATCH_SIZE_VALIDATE)
    c.set_classes(['sans', 'avec'])
    
    if not(MODELE_ENREGISTRE):
        c.train('TRAIN', 'TEST', N_EPOCHS)
    
    if ENREGISTRER_MODELE:
        c.save(ROOT_DIR + 'rsnt18.pth')
    
    # tc, s, e = c.validate('VALIDATION', 5)
    
    # print("Nombre d'images : ",tc)
    # print()
    # print("Détection de l'absence de ligne :")
    # print("   Erreurs :", e[0])
    # print("   Réussites :", s[0])
    # print()
    # print("Détection de la présence de ligne :")
    # print("   Erreurs :", e[1])
    # print("   Réussites :", s[1])
    
    test = c.test_image(ROOT_DIR + "validation_dataset/class0/1431_cam_image_array_.jpg")
    fig = plt.figure(0)
    fig.suptitle(test[1]) 
    plt.imshow(test[0])
    plt.show()
