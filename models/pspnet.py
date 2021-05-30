import torch 
import torch.nn as nn
class convBlock(nn.Module):
	def __init__(self, in_filter, filters):
		super(convBlock, self).__init__()
		f1, f2, f3 = filters

		self.conv = nn.Sequential(
			nn.Conv2d(in_filter, f1, kernel_size=1, padding=0),
			nn.BatchNorm2d(f1),
			nn.LeakyReLU(0.2),

			nn.Conv2d(f1, f2, kernel_size=3, padding=1),
			nn.BatchNorm2d(f2),
			nn.LeakyReLU(0.2),


			nn.Conv2d(f2, f3, kernel_size=1, padding=0),
			nn.BatchNorm2d(f3),
			nn.LeakyReLU(0.2),

			)
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_filter, f3, kernel_size=3, padding=1),
			nn.BatchNorm2d(f3),
			)
	def forward(self, x):
		self.x = x
		self.x_skip = x

		self.x = self.conv(self.x)
		self.x_skip = self.conv2(self.x_skip)

		self.x = torch.add(self.x, self.x_skip)
		self.x = nn.ReLU()(self.x)

		return self.x


class baseFeatureMaps(nn.Module):
	def __init__(self, batch_size):
		super(baseFeatureMaps, self).__init__()
		self.batch_size = batch_size

		self.base = nn.Sequential(
			convBlock(in_filter=3,filters=[32, 32, 64]),
			convBlock(in_filter=64, filters=[64, 64, 128]),
			convBlock(in_filter=128, filters=[128, 128, 256])
			)

	def forward(self, x):
		# self.output = self.base(x)
		# return self.output

		return self.base(x)

class GlobalAvg(nn.Module):
	def __init__(self, batch_size, filters_size):
		super(GlobalAvg, self).__init__()
		self.batch_size = batch_size
		self.filters_size = filters_size
	def forward(self, x):
		return torch.mean(x.view(self.batch_size, self.filters_size, -1), dim=2).view(self.batch_size, self.filters_size, 1, 1)

		
class PSPNET(nn.Module):
	def __init__(self, batch_size, output_size, num_classes):
		super(PSPNET, self).__init__()

		self.num_classes = num_classes
		if self.num_classes == 2:
			self.num_classes -= 1
			self.lastLayer=nn.Sigmoid()
		else:
			self.lastLayer=nn.Softmax(dim=1)

		if (output_size % 8) != 0:
			raise TypeError("PSPNET require Multiple of 8 as a input Size")

		self.base = baseFeatureMaps(batch_size)

		self.red = nn.Sequential(
			GlobalAvg(batch_size, 256),
			nn.Conv2d(256, 64, kernel_size=1),
			nn.Upsample(scale_factor=output_size, mode='bilinear')
			)

		self.yellow = nn.Sequential(
			nn.AvgPool2d(kernel_size=(2, 2)),
			nn.Conv2d(256, 64, kernel_size=1),
			nn.Upsample(scale_factor=2, mode='bilinear')
			)

		self.blue = nn.Sequential(
			nn.AvgPool2d(kernel_size=(4, 4)),
			nn.Conv2d(256, 64, kernel_size=1),
			nn.Upsample(scale_factor=4, mode='bilinear')
			)

		self.green = nn.Sequential(
			nn.AvgPool2d(kernel_size=(8, 8)),
			nn.Conv2d(256, 64, kernel_size=1),
			nn.Upsample(scale_factor=8, mode='bilinear')
			)

		self.final = nn.Sequential(
			nn.Conv2d(512, self.num_classes, kernel_size=3, padding=1),
			nn.BatchNorm2d(self.num_classes)
			)
		
	def forward(self, x):
		self.x = self.base(x)

		self.output1 = self.red(self.x)

		self.output2 = self.yellow(self.x)

		self.output3 = self.blue(self.x)

		self.output4 = self.green(self.x)

		# print(torch.cat((self.output1, self.output2, self.output3, self.output4), 1).shape)
		# print(self.x.shape, self.output1.shape, self.output2.shape, self.output3.shape, self.output4.shape)
		self.x = torch.cat((self.x, self.output1, self.output2, self.output3, self.output4), 1)

		self.x = self.final(self.x)
		self.x = self.lastLayer(self.x)
		return self.x



if __name__ == "__main__":
	batch_size = 2
	input_size = 320

	# optimizers Adam,SGD,Nadam with learning rates [ 0.001, 0.001 , 0.01]
	# OpenCV onnx infernce possible
	model = PSPNET(batch_size=2, output_size=input_size, num_classes=3)

	# print(model)
	# torch.save(model, "pspnet.pth")
	print(model(torch.randn(batch_size, 3, input_size, input_size)).shape)