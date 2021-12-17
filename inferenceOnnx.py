import cv2, os
import numpy as np
from PIL import Image


modelPath = "check_points/unet.onnx"
width = 256
height = 256

net = cv2.dnn.readNetFromONNX(modelPath)


imagePath = r"IMG_6574.JPG"

image = np.array(Image.open(imagePath))


h, w = image.shape[:2]
image_copy = cv2.resize(cv2.cvtColor(image.copy(), cv2.COLOR_RGBA2BGR), (width, height))
print(image_copy.max(), image_copy.min())


image = image.astype(np.float32)

blob = cv2.dnn.blobFromImage(image, size=(width, height), scalefactor=1/127.5, mean=(104, 117, 123))
print(blob.max(), blob.min())
net.setInput(blob)
mask = net.forward()
mask = np.round(mask).astype(np.uint8) * 255
print(np.unique(mask), mask.shape)
mask = np.moveaxis(mask, 1, -1).reshape(height, width)

cv2.imshow("image_copy", image_copy)
cv2.imshow("mask", mask)
# cv2.imshow("image", cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_RGBA2BGR))

print(image_copy.shape, mask.shape, image_copy.dtype, mask.dtype)
result = cv2.bitwise_and(image_copy, image_copy, mask=mask)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.imwrite(f"images/{os.path.split(imagePath)[-1]}", cv2.resize(image_copy, (w, h)))
cv2.imwrite(f"images/output_{os.path.split(imagePath)[-1]}", cv2.resize(result, (w, h)))
cv2.imwrite(f"images/mask_{os.path.split(imagePath)[-1]}", cv2.resize(mask, (w, h)))
