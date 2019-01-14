from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()  

def check():
	use_gpu = torch.cuda.is_available()
	#if use_gpu:
	#    print("Using CUDA")
	# Load the pretrained model from pytorch
	vgg16 = models.vgg16_bn()
	#vgg16.load_state_dict(torch.load("/home/aman/hackathons/GE hack/fastai/courses/dl1/vgg16_bn-6c64b313.pth"))
	#print(vgg16.classifier[6].out_features) # 1000 


	# Freeze training for all layers
	for param in vgg16.features.parameters():
		param.require_grad = False

	# Newly created modules have require_grad=True by default
	num_features = vgg16.classifier[6].in_features
	features = list(vgg16.classifier.children())[:-1] # Remove last layer
	features.extend([nn.Linear(num_features, 4)]) # Add our layer with 4 outputs
	vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
	#print(vgg16)
	resume_training = False

	if resume_training:
		print("Loading pretrained model..")
		vgg16.load_state_dict(torch.load('../input/vgg16-transfer-learning-pytorch/VGG16_v2-OCT_Retina.pt'))
		print("Loaded!")
	if use_gpu:
	 	vgg16.cuda() #.cuda() will move everything to the GPU side
	    
	criterion = nn.CrossEntropyLoss()

	optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
	#vgg16 = models.vgg16_bn()
	vgg16.load_state_dict(torch.load('/home/aman/hackathons/GE hack/fastai/courses/dl1/VGG16_v2-OCT_Retina_half_dataset.pt'))
	from PIL import Image
	imsize = 256
	loader = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),])

	def image_loader(image_name):
		"""load image, returns cuda tensor"""
		image = Image.open(image_name)
		image = loader(image).float()
		image = Variable(image, requires_grad=False)
		image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
		return image.cuda()  #assumes that you're using GPU

	image = image_loader('NORMAL-12494-6.jpeg')
	vgg16.eval()
	outputs=vgg16(image)
	a, preds = torch.max(outputs.data, 1)
	ps = torch.exp(outputs)
	predicted_labels = [preds[j] for j in range(image.size()[0])]
	sm = torch.nn.Softmax()
	probabilities = sm(outputs) 
	print(probabilities)
check()