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

	dataset = Dataset(imageDir=args['images'], maskDir=args['masks'], 
		imageSize=int(args['size']), oneHot=oneHot, numClasses=int(args['classes']))

	dataloader = data.DataLoader(dataset, batch_size=args['batch_size'], shuffle=True)

	for j in range(int(args['epochs'])):

		torch.cuda.empty_cache()
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

				runningLoss += loss.item()

				# tepoch.set_postfix(loss=loss.item(), accuracy=100. * train_acc)
				tepoch.set_postfix(loss=loss.item())
				time.sleep(0.005)

				if i == len(dataloader) - 1:
					# accuracy = accuracy / len(dataloader)
					# tepoch.set_postfix(loss=runningLoss/len(dataloader), accuracy=100. * accuracy)
					tepoch.set_postfix(loss=runningLoss/len(dataloader))

	torch.save(model, "final_model.pth")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--data", required=True, help="Information Related to train process")
	args = vars(parser.parse_args())
	with open(args['data'], "r") as f:
		args = eval(f.read())
	args['images'] = os.path.join(args['train_folder'], "images").replace("\\", "/")
	args['masks'] = os.path.join(args['train_folder'], "masks").replace("\\", "/")
	print(args)
	train(args)
