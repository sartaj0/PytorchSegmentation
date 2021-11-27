# from torch.utils.data import Dataset, DataLoader

import os
import numpy as np 
from PIL import Image

import torch
from torch.utils import data 
from torchvision import transforms

import sys

np.set_printoptions(threshold=sys.maxsize)

class Dataset(data.Dataset):#
	def __init__(self, imageDir, maskDir, imageSize, numClasses, oneHot=False):
		self.imageSize = imageSize
		endswith = (".jpg", ".JPG", ".png", ".jpeg", ".webp")

		self.imagePaths = []
		self.maskPaths = []

		masks = os.listdir(maskDir)
		for image in os.listdir(imageDir):
			if not image.endswith(endswith):
				continue
			j = image.rindex(".")
			for ends in endswith:
				if os.path.isfile(os.path.join(maskDir, image[:j]+ends)):
					self.imagePaths.append(os.path.join(imageDir, image))
					self.maskPaths.append(os.path.join(maskDir, image[:j]+ends))
					break


		self.transform = transform = transforms.Compose([
			transforms.Resize((imageSize, imageSize)),
			transforms.ToTensor(), 
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			
			])

		self.transform2 = transforms.Compose([transforms.Resize((imageSize, imageSize))])

		# print(len(self.imagePaths), len(self.maskPaths))

		self.oneHot = oneHot
		self.numClasses = numClasses


	def __len__(self):
		return len(self.imagePaths)


	def __getitem__(self, idx):
		image = Image.open(self.imagePaths[idx])
		image = self.transform(image)


		mask = Image.open(self.maskPaths[idx])
		mask = self.transform2(mask)

		# mask = torch.tensor(mask)
		mask = np.array(mask)
		if self.oneHot:
			mask = one_hot_targets = np.eye(self.numClasses)[mask]
		else:
			mask = mask.reshape(self.imageSize, self.imageSize, 1)
		mask = torch.from_numpy(mask).type(torch.float32)
		mask = mask.permute(2, 0, 1)

		return image, mask



if __name__ == "__main__":
	dataset = Dataset(imageDir=r'E:\dataset\segmentation\Person\people_segmentation\images',
		maskDir=r'E:\dataset\segmentation\Person\people_segmentation\masks', imageSize=224,
		oneHot=True, numClasses=2)

	dataloader = data.DataLoader(dataset, batch_size=50, shuffle=True)

	for i, (image, mask) in enumerate(dataloader):
		print(i, image.shape, image.dtype, mask.shape, mask.dtype)