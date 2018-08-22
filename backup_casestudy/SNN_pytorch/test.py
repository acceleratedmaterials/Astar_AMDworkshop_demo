import torchvision
import torchvision.utils
import numpy as np
import random
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from model import *
from dataset import *
from hyperparameter import *

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)
										  
#Hyper parameter
net = SiameseNetwork()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = learning_rate)

counter = []
loss_history = [] 
iteration_number= 0


for epoch in range(0, n_epoch):
	for i, data in enumerate(getTrainData(),0):
		x1, x2 , label = data
		print(x1)
		optimizer.zero_grad()
		output1, output2 = net(x1, x2)
		loss_contrastive = criterion(output1, output2, label)
		loss_contrastive.backward()
		optimizer.step()
		if i %10 == 0 :
		    print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
		    iteration_number +=10
		    counter.append(iteration_number)
		    loss_history.append(loss_contrastive.item())
