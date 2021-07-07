# python train.py -d data.json

import torch
import torchvision
import argparse 
from torchvision import transforms
import os 
from PIL import Image
import numpy as np 
from models.pspnet import PSPNET
from loss.dice_loss import DiceLoss, DiceBCELoss
from dataloader.dataloader import Dataset
from torch.utils import data 
from torch import optim
from tqdm import tqdm
import time
torch.cuda.empty_cache()


def train(args):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	batch_size = 1

	if args['model'] == 'pspnet':
		model = PSPNET(output_size=int(args['size']), num_classes=int(args['classes']))

	elif args['model'] == 'segnet':
		pass 
	else:
		raise TypeError("Enter Valid Model Name")
	if os.path.isfile(args["dataset_name"]+".pth"):
		answer = input("Model is already exist would you like to load the weights (y/n): ")
		if (answer.lower() == "y") or (answer.lower() == "yes"):
			model.load_state_dict(torch.load(args["dataset_name"]+".pth"))

	model.to(device)



	if args['loss'] == 'dice':
		criterion = DiceLoss()
	elif args['loss'] == 'dicebce':
		criterion = DiceBCELoss()
	elif args['loss'] == 'binary':
		criterion = nn.BCELoss()
	else:
		raise TypeError("Enter Valid Loss Name")

	learning_rate = 0.001
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	oneHot = True
	if int(args['classes']) == 2:
		oneHot = False

	trainDataset = Dataset(imageDir=args['train_images'], maskDir=args['train_masks'], 
		imageSize=int(args['size']), oneHot=oneHot, numClasses=int(args['classes']))

	dataloader = data.DataLoader(trainDataset, batch_size=args['batch_size'], shuffle=True, num_workers = 3)



	if args['validation']:
		testDataset = Dataset(imageDir=args['val_images'], maskDir=args['val_masks'], 
			imageSize=int(args['size']), oneHot=oneHot, numClasses=int(args['classes']))

		testDataLoader = data.DataLoader(testDataset, batch_size=args['batch_size'], shuffle=True, num_workers = 3)



	for j in range(int(args['epochs'])):

		torch.cuda.empty_cache()

		model.train()
		with tqdm(dataloader, unit="batch") as tepoch:
			runningLoss = 0
			torch.cuda.empty_cache()
			for i, trainingData in enumerate(tepoch):

				(image, mask) = trainingData
				tepoch.set_description(f"Training Epoch {j + 1}")
				image, mask = image.to(device), mask.to(device)
				# torch.cuda.empty_cache()

				optimizer.zero_grad()

				output = model.forward(image)

				loss = criterion(output, mask)

				loss.backward()
				optimizer.step()

				loss_value = loss.item()
				runningLoss += loss_value

				# tepoch.set_postfix(loss=loss.item(), accuracy=100. * train_acc)
				tepoch.set_postfix(loss=loss_value)
				time.sleep(0.005)

				if i == len(dataloader) - 1:
					# accuracy = accuracy / len(dataloader)
					# tepoch.set_postfix(loss=runningLoss/len(dataloader), accuracy=100. * accuracy)
					tepoch.set_postfix(loss=runningLoss/len(dataloader))



		# Validation 
		runningLoss = 0
		model.eval()
		if args['validation']:
			with torch.no_grad():
				with tqdm(testDataLoader, unit="batch") as tepoch:
					torch.cuda.empty_cache()
					for i, testingData in enumerate(tepoch):
						(image, mask) = testingData
						tepoch.set_description(f"Testing Epoch {j + 1}")
						image, mask = image.to(device), mask.to(device)

						output = model.forward(image)

						loss = criterion(output, mask)

						loss_value = loss.item()
						runningLoss += loss_value

						# tepoch.set_postfix(loss=loss.item(), accuracy=100. * train_acc)
						tepoch.set_postfix(loss=loss_value)
						time.sleep(0.005)

						if i == len(testDataLoader) - 1:
							# accuracy = accuracy / len(testDataLoader)
							# tepoch.set_postfix(loss=runningLoss/len(testDataLoader), accuracy=100. * accuracy)
							tepoch.set_postfix(loss=runningLoss/len(testDataLoader))




	# torch.save(model, "final_model.pth")
	torch.save(model.state_dict(), args["dataset_name"]+".pth")


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
