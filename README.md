# PytorchSegmentation


## Dependency 
- pytorch 1.8.1, CUDA 10.2 <br> `pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102` 
- Requirements: Pillow, opencv-python, tqdm, matplotlib <br> `pip3 install Pillow opencv-python tqdm matplotlib` 


## Onnx Model
From [here](https://drive.google.com/drive/folders/15VQZVkfNaAGdoFVUabpdWid6aUw_i7d_?usp=sharing) you can download onnx converted model and use this [file](https://github.com/sartaj0/PytorchSegmentation/blob/main/inferenceOnnx.py) for inferencing with opencv. 


## Output Example
| Image | Prediction Mask | Output |
| --- | --- | --- |
|<img src="./images/IMG_6598.JPG" width="250" title="failure cases"> | <img src="./images/mask_IMG_6598.JPG"  width="250" title="failure cases"> | <img src="./images/output_IMG_6598.JPG"  width="250" title="failure cases"> |
|<img src="./images/IMG_7124.JPG" width="250" title="failure cases"> | <img src="./images/mask_IMG_7124.JPG"  width="250" title="failure cases"> | <img src="./images/output_IMG_7124.JPG"  width="250" title="failure cases"> |
|<img src="./images/IMG_7111.JPG" width="250" title="failure cases"> | <img src="./images/mask_IMG_7111.JPG"  width="250" title="failure cases"> | <img src="./images/output_IMG_7111.JPG"  width="250" title="failure cases"> |
|<img src="./images/IMG_7119.JPG" width="250" title="failure cases"> | <img src="./images/mask_IMG_7119.JPG"  width="250" title="failure cases"> | <img src="./images/output_IMG_7119.JPG"  width="250" title="failure cases"> |