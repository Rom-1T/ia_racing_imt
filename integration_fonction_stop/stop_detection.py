import torch
from torchvision.transforms import transforms

class StopDetection():
    
    def __init__(self, cfg):
        self.device = cfg.STOP_DETECTION_DEVICE
        self.load_model(cfg.STOP_DETECTION_MODEL_PATH)
        
        self.lap_counter = 0
        self.lap_counter_max = cfg.LAP_COUNTER_MAX - 1
        
        self.number_previous_images = cfg.STOP_DETECTION_PREVIOUS_IMG_BASE
        self.previous_image_labels = [None for _ in range(self.number_previous_images)]
        
        self.end = False
        
        self.prints = cfg.STOP_DETECTION_PRINT
        
    def load_model(self, model_pth: str):
        self.model = torch.load(model_pth)
    
    def run(self, image_arr, throttle):
        print("Tour", self.lap_counter)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        image = transform(image_arr)
        image = torch.unsqueeze(image, 0)
        
        with torch.no_grad():
            outputs = self.model(image.to(self.device))
            output_label = int(torch.topk(outputs, 1).indices)
        
        """
            Si on ne detecte pas de ligne :
            - on est dans la course : pas d'action
                - Alors l'image d'avant n'avait pas de ligne (self.previous_image_label := 0)
            - on vient de passer la ligne :
                - Alors l'image d'avant etait une ligne (self.previous_image_label := 1)
        """
        
        if output_label == 0:
            if all(self.previous_image_labels):
                self.lap_counter += 1
                
                if self.prints:
                    print("Franchissement de ligne")
        else:
            if self.prints:
                print('Detection de ligne')
        
        self.previous_image_labels = self.previous_image_labels[1: self.number_previous_images - 1] + [output_label]
        
        if self.end:
            return 0, self.lap_counter, True
        
        if self.trigger_stop():
            self.end = True
            
            if self.prints:
                print("Arrivee passee ! ")
                
            return -1.1*throttle, self.lap_counter, True # Pour que l'arret soit visible
    
        return throttle, self.lap_counter, False
        
    def trigger_stop(self):
        return self.lap_counter > self.lap_counter_max
        