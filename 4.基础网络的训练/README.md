#### 基础网络的训练

目前我们常用的神经网络，github上基本都具有较为丰富的训练、测试代码，我们这里选择几种常用，高效的网络推荐给大家，包括与之对应的github工程，涉及一些训练的技巧，旨在让大家能够复现出作者原始的精度。

我们这里主要介绍以下几个网络的训练与使用:

```
1.MobileNet(分类网络)

2.MnasNet(分类网络)

3.MTCNN(单一物体检测网络)

4.MobileNet-SSD(Single Shot 物体检测网络)

5.CTPN(文字定位网络)

6.insightface(人脸识别网络)

7.VanillaCNN(人脸关键点回归网络)

8.YOLO-V3(通用物体检测网络)

9.DeepOCR(文字识别网络)

```

以上这些网络涵盖了日常使用网络设计到的大部分功能，一些相关的应用也可以通过这些网络的变通，修改进行试验。

#### 1.MobileNet

MobileNet是谷歌发布的第一代专为移动端设计的高效网络，其后续版本MobileNet-v2同样优秀，shicai yang大神已经给出了网络的pretrain model，以及caffe的[训练、测试代码](https://github.com/shicai/MobileNet-Caffe)，利用该网络可以训练其他类似的分类任务，例如我们开源的[鉴黄网络](https://github.com/zeusees/HyperNSFW).

#### 2.MnasNet

MnasNet同样是谷歌发布的高效移动端分类网络，与Mobilenet不同之处在于网络的设计借助deepmind AI的能力，不是hand craft手动设计的网络，相比于mobilenet，速度快大约1.5倍，准确度提高将近两个点。我们同样复现了该网络，并且提供了该网络再标准ImageNet上的pretrain model，接近了官方的精度。连接地址：https://github.com/zeusees/Mnasnet-Pretrained-Model

#### 3.MTCNN

MTCNN是一个非常优秀的单一物体检测框架，可以用这个框架进行人脸、车辆、行人等单一物体的检测，该网络的主要问题在于单帧图像中包含多个物体时，检测速度下降严重。mtcnn的复现在github上有多个版本，包括caffe、keras、TensorFlow等，我们测试了不同版本，有一些存在问题，https://github.com/AITTSMD/MTCNN-Tensorflow 这个repo能够基本复现作者的精度，训练过程中，一定要注意正负样本保持1:3的比例。其实，mtcnn框架具有一些优化的方法和空间，包括用卷积替代polling，采用dw卷积等等，相关修改可以参考我们的文章: https://blog.csdn.net/Relocy/article/details/84075570 . 我们的工程师同样提供了一个优化的mtcnn模型：https://github.com/szad670401/Fast-MTCNN ，大家可以参考修改。

#### 4.MobileNet-SSD

SSD是Single Shot检测网络的代表结构，其速度快，单帧物体数量对检测速度影响不大，具有很好的工程化指导作用。Mobilenet跟SSD的结合，更能够提高网络的速度。Mobilenet-SSD可以参考：https://github.com/chuanqi305/MobileNet-SSD 这里有数据准备代码，以及网络的训练测试代码。我们采用这个网络进行了车牌检测的实验，效果也不错，能够完成单层、双层、蓝牌、黄牌、绿牌的检测，可以参见我们的博客：https://blog.csdn.net/lsy17096535/article/details/78687728 ，我们开源的车牌检测Mobilenet-SSD模型：https://github.com/zeusees/Mobilenet-SSD-License-Plate-Detection

#### 5.CTPN

#### 6.insightface

insightface是一款高精度的开源人脸识别框架，在我们的测试中，insightface针对一般场景效果不错，“历史脸”效果稍差，有可能因为训练数据历史脸数据不足导致的，算法的作者guojia也将论文提交到了CVPR2019，期待他的好消息。大家可以在这里找到作者的实现：https://github.com/deepinsight/insightface 作者的框架基于MXNET，目前git上已经有基于TensorFlow、caffe等其他框架的实现，大家可以参考。大家在部署阶段，可以利用TVM部署该框架，速度快，也可以将模型转换到caffe model，部署到其他平台。TVM部署方法可以参考我们的博客：[insightface模型的TVM框架部署](https://github.com/zeusees/HyperDL-Tutorial/blob/master/5.%E6%A8%A1%E5%9E%8B%E7%9A%84%E9%83%A8%E7%BD%B2/%E5%9C%A8CPP%E4%B8%8B%E4%BD%BF%E7%94%A8TVM%E6%9D%A5%E9%83%A8%E7%BD%B2mxnet%E6%A8%A1%E5%9E%8B%EF%BC%88%E4%BB%A5Insightface%E4%B8%BA%E4%BE%8B%EF%BC%89.md)

#### 7.VanillaCNN

VanillaCNN是针对香港中文大学人脸关键点定位网络TCDCN的一个复现，大家可以参考 https://github.com/ishay2b/VanillaCNN 。稠密人脸关键点定位(通常关键点50点以上)同样是一个回归问题，让网络能够通过对人脸边缘特征的提取，回归出准确的定位，这篇文章采用了多任务进行定位，取得了很好的效果，后来几年的的很多算法，在准确度上有提升，但是在速度上不具有优势。大家可以利用高效的网络结构提取特征并加速，取得更好的人脸关键点定位准确度和速度。大家在训练关键点定位的网络时，可以结合可视化的技术，将网络后面基层的feature map显示出来，观察网络对输入人脸边缘提取的效果，改进网络结构。

#### 8.YOLO-V3

通用物体检测近年来也是研究人员关注的人们领域，从RBG、何凯明大神的RCNN，Fast RCNN，Faster RCNN，MASK RCNN等，Single Shot的Yolo系列、SSD等，以后后来的RetinaNet，我们对这一系列的网络都进行过测试，由于我们算法组在日常使用中主要考虑移动端的部署以及服务器端的效率，推荐了MobileNet-SSD跟YOLO-V3。我们对3000张行车记录仪标注图像以及2000张交通监控图片进行标注，分别在以上网络进行了测试，对于我们的图片，YOLO-V3表现最好，速度也是最快的一档。项目主页：https://pjreddie.com/darknet/yolo/ 

#### 9.DeepOCR

DeepOCR这里我们泛指利用深度学习进行文字识别的方法，目前网上开源的算法较好的有 https://github.com/YCG09/chinese_ocr 以及 https://github.com/chineseocr/chineseocr ， 前一个方法利用CTPN网络作为文字检测网络，然后利用DenseNet + CTC 进行文字识别。后面的方法利用YOLOv3作文文字检测网络，利用CRNN进行文字识别。两个网络都提供了训练代码，直接使用已有的模型对黑白打印文字识别效果还可以，但是对于自然文字场景，文字颜色不为黑色的情况下，识别率较差，使用者可以利用 https://github.com/yanhaiming56/SynthText_Chinese_py3 这个工具生成自己需要的字体及颜色的样本（我们观察了一下，生成的样本质量一般），或者自己标注新的数据，重新训练模型，取得好的效果。
