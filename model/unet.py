
import torch
import torch.nn as nn
from torch.nn import functional as F

class Block(nn.Module):
	def __init__(self, inChannels, outChannels):
		super(Block, self).__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(inChannels, outChannels, 3, padding=1),
			nn.BatchNorm2d(outChannels),
			nn.ReLU(),

			nn.Conv2d(outChannels, outChannels, 3, padding=1),
			nn.BatchNorm2d(outChannels),
			nn.ReLU()
			)

	def forward(self, x):
		return self.conv(x)

class Encoder(nn.Module):
	def __init__(self, channels):
		
		super(Encoder, self).__init__()

		self.encBlock = [Block(channels[i], channels[i+1]) for i in range(len(channels) - 1)]
		self.pool = nn.MaxPool2d(2)

		self.encBlock = nn.ModuleList(self.encBlock)

	def forward(self, x):
		output = []
		for block in self.encBlock:
			x = block(x)
			output.append(x)
			x = self.pool(x)
		return output


class Decoder(nn.Module):
	def __init__(self, channels):
		super(Decoder, self).__init__()
		self.channels = channels

		# self.upconvs = [nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)]
		self.upconvs = [self.ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)]
		self.upconvs = nn.ModuleList(self.upconvs)

		self.decBlocks = [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
		self.decBlocks = nn.ModuleList(self.decBlocks)


	def forward(self, x, encFeatures):
		for i in range(len(self.channels) - 1):
			x = self.upconvs[i](x)

			encFeat = self.crop(encFeatures[i], x)
			x = torch.cat([x, encFeat], dim=1)

			x = self.decBlocks[i](x)

		return x

	def crop(self, encFeatures, x):
		(_, _, h1, w1) = x.shape
		(_, _, h2, w2) = encFeatures.shape

		hdist = int((h2 - h1) / 2)
		wdist= int((w2 - w1) / 2)

		encFeatures = encFeatures[:, :, hdist: hdist + h1, wdist: wdist + w1]

		return encFeatures

	def ConvTranspose2d(self, channel1, channel2, kernel, stride):
		return nn.Sequential(
			nn.ConvTranspose2d(channel1, channel2, kernel, stride),
			nn.BatchNorm2d(channel2),
			nn.ReLU()
			)

class UNet(nn.Module):
	def __init__(self, channels, num_classes=2, outputSize=256, retainDim=False):

		super(UNet, self).__init__()

		encChannels = channels
		decChannels = channels[1:][::-1]

		self.num_classes = num_classes
		if self.num_classes == 2:
			self.num_classes -= 1
			self.lastLayer=nn.Sigmoid()
		else:
			self.lastLayer=nn.Softmax(dim=1)

		self.encoder = Encoder(encChannels)
		self.decoder = Decoder(decChannels)

		self.head = nn.Sequential(
			nn.Conv2d(decChannels[-1],  self.num_classes, 1),
			self.lastLayer
			)
		self.retainDim = retainDim
		self.outputSize = outputSize

	def forward(self, x):
		encFeatures = self.encoder(x)
		decFeatures = self.decoder(encFeatures[::-1][0], encFeatures[::-1][1:])

		fmap = self.head(decFeatures)

		if self.retainDim:
			fmap = F.interpolate(fmap, (self.outputSize, self.outputSize))

		return fmap

if __name__ == '__main__':
	channels = [3, 32, 64, 128, 256, 512]

	# enc = Encoder(channels)
	# dec = Decoder(channels[1:][::-1])
	# print(enc)
	# print(dec)
	# a = torch.rand([1, 3, 512, 512])
	# b = enc(a)
	# b = b[::-1]
	# c = dec(b[0], b[1:])
	# print(c.shape)

	outputSize = 416
	unet = UNet(channels, retainDim=True, outputSize=416)
	print(unet)
	a = torch.rand([1, 3, outputSize, outputSize])
	print(unet(a).shape)
	torch.save(unet.state_dict(), "unet.pth", )
	