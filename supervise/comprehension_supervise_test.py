# Load and run neural network and make preidction
import numpy as np
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf

width = 900
height = 900 # image width and height
modelPath="3000.torch"
#---------------------create image ---------------------------------------------------------

FillLevel=0.7
Img=np.zeros([width,height,3],np.uint8)
Img[0:int(FillLevel*height),:]=255
#-------------Transform image to pytorch------------------------------------------------------
transformImg = tf.Compose([tf.ToPILImage(), tf.Resize((height, width)), tf.ToTensor(),
                            tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

Img=transformImg(Img)

#------------------Create batch----------------------------------------------
images = torch.zeros([1,3,height,width]) # Create batch for one image
images[0]=Img

#--------------Load and  net-------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
Net = torchvision.models.resnet18(pretrained=True) # Load net
Net.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True) # Set final layer to predict one value
Net = Net.to(device) # Assign net to gpu or cpu
#
Net.load_state_dict(torch.load(modelPath)) # Load trained model
#Net.eval() # Set net to evaluation mode, usually usefull in this case its fail
#----------------Make preidction--------------------------------------------------------------------------
Img = torch.autograd.Variable(Img, requires_grad=False).to(device).unsqueeze(0) # Convert to pytorch
with torch.no_grad():
    Prd = Net(Img)  # Run net
print("Predicted fill level", float(Prd))
print("Real fill level", FillLevel)