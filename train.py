# python train.py -i E:\dataset\segmentation\Person\people_segmentation\images -m E:\dataset\segmentation\Person\people_segmentation\masks -s 320 -nc 2 -mo pspnet -l dice -e 100

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

torch.cuda.empty_cache()

def train(args):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	batch_size = 1

	if args['model'] == 'pspnet':
		model = PSPNET(output_size=int(args['size']), num_classes=int(args['noc']))

	elif args['model'] == 'segnet':
		pass 
	else:
		raise TypeError("Enter Valid Model Name")
	model.to(device)


	if args['loss'] == 'dice':
		criterion = DiceLoss()
	elif args['loss'] == 'dicebce':
		criterion = DiceBCELoss()
	else:
		raise TypeError("Enter Valid Loss Name")

	learning_rate = 0.001
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	oneHot = True
	if int(args['noc']) == 2:
		oneHot = False

	dataset = Dataset(imageDir=args['images'], maskDir=args['masks'], 
		imageSize=int(args['size']), oneHot=oneHot, numClasses=int(args['noc']))

	dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

	for j in range(int(args['epochs'])):
		runningLoss = []
		torch.cuda.empty_cache()
		for i, (image, mask) in enumerate(dataloader):
			image, mask = image.to(device), mask.to(device)
			# torch.cuda.empty_cache()

			optimizer.zero_grad()

			output = model.forward(image)

			loss = criterion(output, mask)

			loss.backward()
			optimizer.step()

			runningLoss.append(loss.item())

			print(f"Epochs{j+1}, Iteration: {i+1}, Loss:",loss.item())

		print("Loss:", sum(runningLoss) / len(runningLoss))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--images", required=True, help="Images Folder")
	parser.add_argument("-m", "--masks", required=True, help="Mask Folder")
	parser.add_argument("-s", "--size", required=True, help="Image Size")
	parser.add_argument("-nc", "--noc", required=True, help="Number of Classes")
	parser.add_argument("-mo", "--model", required=True, help="Model Name e.g. pspnet, unet, enet etc.")
	parser.add_argument("-l", "--loss", required=True, help="Loss Name e.g. dice, dicebce, focal etc.")
	parser.add_argument("-e", "--epochs", required=True, help="Number of Epochs ")
	args = vars(parser.parse_args())

	print(args)
	train(args)
