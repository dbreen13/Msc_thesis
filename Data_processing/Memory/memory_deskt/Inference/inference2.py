#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:40:19 2024

@author: dbreen

import torchvision

"""

import torchvision
import torchvision.datasets as datasets
import logging
import torch
import torchvision.transforms as transforms
import time as timers

from time import time, perf_counter
import time as timers
from datetime import datetime
from tqdm import tqdm

mean_CIFAR10 = [0.49139968, 0.48215841, 0.44653091]
std_CIFAR10 = [0.24703223, 0.24348513, 0.26158784]

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean_CIFAR10,
            std_CIFAR10,
        ),
    ])

testset = datasets.CIFAR10(root='/home/dbreen/Documents/tddl/bigdata/cifar10', train=False,download=False, transform=transform)
batch_size=128
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=8)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

logger=logging.getLogger('Inferencefin')
#create a fh
fh=logging.FileHandler('inferencefin.log')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

#baseline
path="/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/baselines/baseline-rn18-cifar10/runnr1/rn18_18_dNone_128_adam_l0.001_g0.1_w0.0_sTrue/cnn_best.pth"   
model=torch.load(path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# since we're not training, we don't need to calculate the gradients for our outputs
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for i in [1,2,3]:
        timers.sleep(60)
        now=datetime.now()
        sec_wait=60-now.second
        timers.sleep(sec_wait)
        print('hello')
        logger.info(f'start-inf-base-cif-ind{i}' )
        for data in testloader:
            images, labels = data
            images = images.to(device)  # Move input data to the same device as the model
            labels = labels.to(device)  # Move labels to the same device as the model
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        logger.info(f'end-inf-base-cif-ind{i}' )
    

    

layers=[63,57,51,47,41,35,28,25,19,6]    
compression=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]    
methods=['cp','tt', 'tucker']

for method in methods:
    for layer in layers:
        for compr in compression:
            path=f"/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-{method}-r{compr}-lay[{layer}]-b128/runnr1/rn18-lr-[{layer}]-{method}-{compr}-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_final.pth"
            model=torch.load(path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            # since we're not training, we don't need to calculate the gradients for our outputs
            correct = 0
            total = 0
            # since we're not training, we do{n't need to calculate the gradients for our outputs
            with torch.no_grad():
                for i in [1,2,3]:
                    timers.sleep(60)
                    now=datetime.now()
                    sec_wait=60-now.second
                    timers.sleep(sec_wait)
    
                    logger.info(f'start-inf-{method}-r{compr}-lay[{layer}]-ind{i}' )
                    t = tqdm(testloader, total=int(len(testloader)))
                    for _ , data in enumerate(testloader):
                        images, labels = data
                        images = images.to(device)  # Move input data to the same device as the model
                        labels = labels.to(device)  # Move labels to the same device as the model
                        # calculate outputs by running images through the network
                        outputs = model(images)
                        # the class with the highest energy is what we choose as prediction
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    logger.info(f'end-inf-{method}-r{compr}-lay[{layer}]' )


