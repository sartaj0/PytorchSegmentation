import torch
import torch.nn as nn
import torch.nn.functional as F

class conBlock2(nn.Module):
	def __init__(self, inFeature, midFeature, outFeature, kernel=3):
		super(conBlock2, self).__init__()

		if kernel % 2 == 0:
			raise TypeError("SegNet Convolution kernel cannot be even")
		self.conv = nn.Sequential(
			nn.Conv2d(inFeature, midFeature, kernel_size=kernel, padding=(kernel - 1) // 2),
			nn.BatchNorm2d(midFeature),
			nn.LeakyReLU(0.2),

			nn.Conv2d(midFeature, outFeature, kernel_size=kernel, padding=(kernel - 1) // 2),
			nn.BatchNorm2d(outFeature),
			nn.LeakyReLU(0.2)
			)
	def forward(self, x):
		return self.conv(x)

class conBlock3(nn.Module):
	def __init__(self, inFeature, midFeature1, midFeature2, outFeature, kernel=3):
		super(conBlock3, self).__init__()

		if kernel % 2 == 0:
			raise TypeError("U-Net Convolution kernel cannot be even")
		self.conv = nn.Sequential(
			nn.Conv2d(inFeature, midFeature1, kernel_size=kernel, padding=(kernel - 1) // 2),
			nn.BatchNorm2d(midFeature1),
			nn.LeakyReLU(0.2),

			nn.Conv2d(midFeature1, midFeature2, kernel_size=kernel, padding=(kernel - 1) // 2),
			nn.BatchNorm2d(midFeature2),
			nn.LeakyReLU(0.2),

			nn.Conv2d(midFeature2, outFeature, kernel_size=kernel, padding=(kernel - 1) // 2),
			nn.BatchNorm2d(outFeature),
			nn.LeakyReLU(0.2)
			)
	def forward(self, x):
		return self.conv(x)

'''
class ArgMaxPooling(nn.Module):
	def __init__(self, pool_size):
		super(ArgMaxPooling, self).__init__()
		self.maxPool = nn.MaxPool2d(pool_size)
	def forward(self, x):
		# self.output = torch.unsqueeze(torch.argmax(x, dim=1), dim=1).type(torch.float32)
		# self.output = self.maxPool(self.output)

		self.output = self.maxPool(x)
		self.output = torch.unsqueeze(torch.argmax(self.output, dim=1), dim=1).type(torch.float32)
		return self.output
'''
class ArgMax(nn.Module):
	def __init__(self):
		super(ArgMax, self).__init__()
	def forward(self, x):

		self.output = torch.unsqueeze(torch.argmax(x, dim=1), dim=1).type(torch.float32)
		return self.output



class SegNet(nn.Module):
	def __init__(self, input_size, n_channels):
		super(SegNet, self).__init__()
		if input_size % 32 != 0:
			raise TypeError("SEGNet require Multiple of 32 as a input Size")

		self.cb1 = conBlock2(3, 64, 64)
		self.argmax = ArgMax()
		self.cb2 = conBlock2(1, 128, 128)
		self.cb3 = conBlock3(1, 256, 256, 256)
		self.cb4 = conBlock3(1, 512, 512, 512)
		self.cb5 = conBlock3(1, 512, 512, 512)

		self.mxpool = nn.MaxPool2d(2, return_indices=True)
		self.mxunpool = nn.MaxUnpool2d(2)

		self.cb6 = conBlock3(1, 512, 512, 512)
		self.cb7 = conBlock3(512, 512, 512, 256)
		self.cb8 = conBlock3(256, 256, 256, 128)
		self.cb9 = conBlock2(128, 128, 64)
		# self.cb10 = conBlock2(64, 64, n_channels)
		self.cb10 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=(3 - 1) // 2),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2),

			nn.Conv2d(64, n_channels, kernel_size=3, padding=(3 - 1) // 2),
			nn.BatchNorm2d(n_channels),
			nn.Sigmoid()
			)

	def forward(self, x):

		x = self.cb1(x)
		x, indices1 = self.mxpool(x)
		x = self.argmax(x)

		x = self.cb2(x)
		x, indices2 = self.mxpool(x)
		x = self.argmax(x)

		x = self.cb3(x)
		x, indices3 = self.mxpool(x)
		x = self.argmax(x)

		x = self.cb4(x)
		x, indices4 = self.mxpool(x)
		x = self.argmax(x)

		x = self.cb5(x)
		x = self.argmax(x)
		x, indices5 = self.mxpool(x)



		x = self.mxunpool(x, indices5)
		x = self.cb6(x)

		x = self.mxunpool(x, indices4)
		x = self.cb7(x)
		
		x = self.mxunpool(x, indices3)
		x = self.cb8(x)

		x = self.mxunpool(x, indices2)
		x = self.cb9(x)
		
		x = self.mxunpool(x, indices1)
		x = self.cb10(x)

		print(x.shape)
		return x
if __name__ == "__main__":
	input_size = 416
	batch_size = 3
	model = SegNet(input_size=224, n_channels=1)
	print(model)
	torch.save(model, "segnet.pth")
	a = torch.rand(batch_size, 3, input_size, input_size)
	model(a)

	#Can't convert to onnx
	# model.eval()
	# dummy_input = torch.randn(1, 3, 416, 416)
	# torch.onnx.export(model, dummy_input, "model.onnx", verbose=True)
