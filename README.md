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
>>#### For testing:<br>
>>Open main.py,change the data path to your data, and excute it.The result will be saved at sample directory.<br>
>>#### For training:<br>
>>Put your own dataset into Train directory,open main.py and turn is_train as Ture.<br>


## Result:
>>After training 100 epochs on 120 images, the networks can be well trained and generate the high resolution image. It takes about 7 hours to train the model,and both training and testing are preformed on single NIVIDA 1080ti GPU.Empirically,i set the initial learning rate 0.0001 and set the decay rate 0.98 with every 1000 steps.the results are shown as following. 
