# from torch.utils.data import Dataset, DataLoader

import os
import numpy as np 
from PIL import Image

import torch
from torch.utils import data 
from torchvision import transforms

import sys

np.set_printoptions(threshold=sys.maxsize)


class Transformation(object):
	def __init__(self, imageSize=False, Normalize=False):
		self.imageSize = imageSize
		self.toTensor = transforms.ToTensor()
		self.normalize = transforms.Normalize(Normalize[0], Normalize[1])

	def Resize(self, image, mask):
		image = image.resize((self.imageSize, self.imageSize))
		mask = mask.resize((self.imageSize, self.imageSize))

		return image, mask

	def HorizontalFlip(self, image, mask):
		image = image.transpose(Image.FLIP_LEFT_RIGHT)
		mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

		return image, mask

	def __call__(self, image, mask):

		# Horizontal Flip
		if np.random.rand() > 0.5:
			image, mask = self.HorizontalFlip(image, mask)
		image, mask = self.Resize(image, mask)

		image = self.toTensor(image)
		image = self.normalize(image)

		return image, mask
		


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


		# self.transform1 = transforms.Compose([transforms.Resize((imageSize, imageSize)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		# self.transform2 = transforms.Compose([transforms.Resize((imageSize, imageSize))])

		self.transform = Transformation(imageSize=imageSize, Normalize=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

		self.oneHot = oneHot
		self.numClasses = numClasses


	def __len__(self):
		return len(self.imagePaths)


	def __getitem__(self, idx):
		image = Image.open(self.imagePaths[idx]).convert("RGB")
		mask = Image.open(self.maskPaths[idx])

		# image = self.transform1(image)
		# mask = self.transform2(mask)

		image, mask = self.transform(image, mask)

		mask = np.array(mask)
		if self.oneHot:
			mask = np.eye(self.numClasses)[mask]
		else:
			mask = mask.reshape(self.imageSize, self.imageSize, 1)
		mask = torch.from_numpy(mask).type(torch.float32)
		
		mask = mask.permute(2, 0, 1)

		return image, mask



if __name__ == "__main__":
	import cv2
	dataset = Dataset(imageDir=r'E:\dataset\segmentation\Person\people_segmentation\images',
		maskDir=r'E:\dataset\segmentation\Person\people_segmentation\masks', imageSize=224,
		oneHot=False, numClasses=2)

	dataloader = data.DataLoader(dataset, batch_size=50, shuffle=True)
	transform = transforms.ToPILImage()


	for i, (image, mask) in enumerate(dataloader):
		print(i, image.shape, image.dtype, mask.shape, mask.dtype)
		for img in image:
			img *= 0.5
			img += 0.5

			img = transform(img)
			img.show()

			mask0 = mask[0].numpy().astype(np.uint8).reshape(224, 224) * 255
			cv2.imshow("mask", mask0)
			cv2.waitKey(0)
			break 
		break