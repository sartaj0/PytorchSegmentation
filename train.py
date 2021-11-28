# python train.py -d data.json

import torch
import torch.nn as nn
from torch import optim
from torch.utils import data 

import torchvision
from torchvision import transforms


import os 
import time
import argparse 
import numpy as np 
from tqdm import tqdm
from PIL import Image

from model.pspnet import PSPNET
from modules.dataloader import Dataset

torch.cuda.empty_cache()


def train(args):

	if not os.path.isdir(args['save_dir']):
		os.mkdir(args['save_dir'])

	args['size'] = int(args['size'])
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	PATH = os.path.join(args['save_dir'], args["model"])

	if args['model'] == 'pspnet':
		model = PSPNET(output_size=int(args['size']), num_classes=int(args['classes']))

	elif args['model'] == 'segnet':
		pass 
	else:
		raise TypeError("Enter Valid Model Name")
	if os.path.isfile(PATH+".pth"):
		answer = input("Model is already exist would you like to load the weights (y/n): ")
		if (answer.lower() == "y") or (answer.lower() == "yes"):
			model.load_state_dict(torch.load(PATH+".pth"))
	print(model)

	model.to(device)

	# '''
	if args['loss'] == 'binary':
		criterion = nn.BCELoss()
	else:
		raise TypeError("Enter Valid Loss Name")

	learning_rate = 0.000087
	optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

	oneHot = True
	if int(args['classes']) == 2:
		oneHot = False

	trainDataset = Dataset(imageDir=args['train_images'], maskDir=args['train_masks'], 
		imageSize=args['size'], oneHot=oneHot, numClasses=int(args['classes']))

	dataloader = data.DataLoader(trainDataset, batch_size=args['batch_size'], shuffle=True)



	if args['validation']:
		testDataset = Dataset(imageDir=args['val_images'], maskDir=args['val_masks'], 
			imageSize=int(args['size']), oneHot=oneHot, numClasses=int(args['classes']))

		testDataLoader = data.DataLoader(testDataset, batch_size=args['batch_size'], shuffle=True)

	minValLoss = None
	
	for epoch in range(1, int(args['epochs']) + 1):

		torch.cuda.empty_cache()

		model.train()
		with tqdm(dataloader, unit="batch", leave=False) as tepoch:
			trainLoss = []
			torch.cuda.empty_cache()
			for i, inputs in enumerate(tepoch):

				tepoch.set_description(f"Training Epoch {epoch}")
				# torch.cuda.empty_cache()

				optimizer.zero_grad()

				output = model.forward(inputs[0].to(device))

				loss = criterion(output, inputs[1].to(device))

				loss.backward()
				optimizer.step()

				loss_value = loss.item()
				trainLoss.append(loss_value)

				tepoch.set_postfix(loss=loss_value)

		# Validation 
		valLoss = []
		model.eval()
		if args['validation']:
			with torch.no_grad():
				with tqdm(testDataLoader, unit="batch", leave=False) as tepoch:
					torch.cuda.empty_cache()
					for i, inputs in enumerate(tepoch):
						# (image, mask) = testingData
						tepoch.set_description(f"Testing Epoch {epoch}")
						# image, mask = image.to(device), mask.to(device)

						output = model.forward(inputs[0].to(device))

						loss = criterion(output, inputs[1].to(device))

						loss_value = loss.item()
						valLoss.append(loss_value)

						# tepoch.set_postfix(loss=loss.item(), accuracy=100. * train_acc)
						tepoch.set_postfix(loss=loss_value)

		print(f"Epochs: {epoch}\t Training Loss: {np.mean(trainLoss)}\t Testing Loss: {np.mean(valLoss)}")

		# '''
		if (minValLoss is None) or (minValLoss > np.mean(valLoss)):
			minValLoss = np.mean(valLoss)
			torch.save(model.state_dict(), PATH+".pth")

		
	model.load_state_dict(torch.load(PATH +".pth"))
	model.to("cpu")
	model.eval()

	dummy_input = torch.randn(1, 3, args['size'], args['size'])
	torch.onnx.export(model, dummy_input,
		os.path.join(f"{PATH}.onnx"), 
		verbose=True)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--data", required=True, help="Information Related to train process")
	args = vars(parser.parse_args())
	with open(args['data'], "r") as f:
		args = eval(f.read())
	args['train_images'] = os.path.join(args['train_folder'], "images").replace("\\", "/")
	args['train_masks'] = os.path.join(args['train_folder'], "masks").replace("\\", "/")

	try:
		if os.path.isdir(args["val_folder"]):
			args['val_images'] = os.path.join(args['val_folder'], "images").replace("\\", "/")
			args['val_masks'] = os.path.join(args['val_folder'], "masks").replace("\\", "/")

			args['validation'] = True
	except Exception as e:
		args['validation'] = False

	print(args)
	train(args)
