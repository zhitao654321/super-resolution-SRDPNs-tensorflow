# super-resolution-SRDPNs-tensorflow
## tensorflow implementation of super-resolution:
>>My implemetaion of super-resolution based on dual-path networks but is quit different from orignal networks,the differences are shown as fllowing:<br>
>>1. I modifed the structure of the dual-path blocks for faster training spped.<br>
>>2. I introduced bottleneck to reduce dimension and deconvolution to restore the detail.<br>
>>3. I introduce the preceptual loss and gram loss based on feature space of VGG19. <br>

## Dependencies:
>>tensorflow >=1.3.0<br>
>>Scipy >= 0.18<br>
>>GPU memory > 7G<br>

## Usage:
>>First, you need to download the module of VGG19 [here](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat) for loss function caulation.<br /> 
>>Then, Move downloaded file
    `imagenet-vgg-verydeep-19.mat `
to SRDPNs folder in this project.<br>
>>#### For testing:<br />
>>Open `main.py`,change the data path to your data,for example:<br>
`flags.DEFINE_string("testimg", "2.bmp", "Name of test image")` <br />
>>Excute the `python main.py` for testing, and The result will be saved at sample directory.<br>
>>#### For training:<br />
>>Put your own dataset into Train directory,change the code<br> as `flags.DEFINE_boolean("is_train",True,"True for training, False for testing")` and excute `python main.py` for training.<br />

## Result:
After training 100 epochs on 120 images, the networks can be well trained and generate the high resolution image. It takes about 7 hours to train the model,and both training and testing are preformed on single NIVIDA 1080ti GPU.Empirically,i set the initial learning rate 0.0001 and set the decay rate 0.98 with every 1000 steps.the results are shown as following. <br />

>>![low-resolution](https://github.com/zhitao654321/super-resolution-SRDPNs-tensorflow/blob/master/sample/ORIGNAL1.png)
>>![result](https://github.com/zhitao654321/super-resolution-SRDPNs-tensorflow/blob/master/sample/test_image1.png)<br>
>>![low-resolution](https://github.com/zhitao654321/super-resolution-SRDPNs-tensorflow/blob/master/sample/ORIGNAL2.png)
>>![result](https://github.com/zhitao654321/super-resolution-SRDPNs-tensorflow/blob/master/sample/test_image2.png)<br>
>>![low-resolution](https://github.com/zhitao654321/super-resolution-SRDPNs-tensorflow/blob/master/sample/ORIGNAL3.png)
>>![result](https://github.com/zhitao654321/super-resolution-SRDPNs-tensorflow/blob/master/sample/test_image3.png)
