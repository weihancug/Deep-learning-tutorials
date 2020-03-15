
## 深度学习经典卷积神经网络

随着深度学习的发展，研究人员提出了很多模型，这其中一些设计方式，在当时取得了很好的效果，也为其他科研工作者提供了很好的思路。CNN 的经典结构始于1998年的LeNet-5，成于2012年历史性的AlexNet，从此广泛用于图像相关领域，主要包括：

    1.LeNet-5, 1998年
    
    2.AlexNet, 2012年
    
    3.ZF-Net, 2013年
    
    4.GoogleNet, 2014年
    
    5.VGG, 2014年
    
    6.ResNet, 2015年

经过科研工作者的反复验证及广泛使用，这些模型逐渐成为经典，我们这里收集了一些常用的模型进行介绍。


### 1. VGG

[论文地址](https://arxiv.org/abs/1409.1556)

VGGNet是牛津大学计算机视觉组（Visual Geometry Group）和 Google DeepMind 公司的研究员一起研发的的深度卷积神经网络。VGGNet 探索了卷积神经网络的深度与其性能之间的关系，一共有六种不同的网络结构，但是每种结构都有含有５组卷积，每组卷积都使用３ｘ３的卷积核，每组卷积后进行一个２ｘ２最大池化，接下来是三个全连接层。在训练高级别的网络时，可以先训练低级别的网络，用前者获得的权重初始化高级别的网络，可以加速网络的收敛。VGGNet 相比之前state-of-the-art的网络结构，错误率大幅下降，并取得了ILSVRC 2014比赛分类项目的第2名和定位项目的第1名。同时VGGNet的拓展性很强，迁移到其他图片数据上的泛化性非常好。VGGNet的结构非常简洁，整个网络都使用了同样大小的卷积核尺寸（3*3）和最大池化尺寸（2*2）。
到目前为止，VGGNet依然经常被用来提取图像特征。

[经典卷积神经网络之VGGNet](https://blog.csdn.net/marsjhao/article/details/72955935)

[VGG模型核心拆解](https://blog.csdn.net/qq_40027052/article/details/79015827)

### 2. GoogLeNet

[[v1] Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842 )

[[v2] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/abs/1502.03167 )

[[v3] Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567) 

[[v4] Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](http://arxiv.org/abs/1602.07261)


GoogLeNet的最早版本，出现在2014年的" Going deeper with convolutions "。之所以名为 "GoogLeNet" 而非 "GoogleNet" ,文章说是为了向早期的LeNet致敬。GoogleNet提出了一个全新的深度 CNN 架构——Inception，无全连接层，可以节省运算的同时，减少了很多参数，参数数量是AlexNet的1/12，数量只有5 million，而且在ImageNet竞赛中取得了很好的成绩。

[GoogleNet系列论文学习](https://blog.csdn.net/cdknight_happy/article/details/79247280)


### 3. Resnet 

[论文地址](https://arxiv.org/abs/1512.03385)

ResNet在2015年被提出，在ImageNet比赛classification任务上获得第一名，因为它 "简单与实用" 并存，之后很多方法都建立在ResNet50或者ResNet101的基础上完成的，检测，分割，识别等领域都纷纷使用ResNet，具有很强的适应性。ResNet的作者[何凯明](http://kaiminghe.com/)也因此摘得CVPR2016最佳论文奖。

[ResNet解析](https://blog.csdn.net/lanran2/article/details/79057994)

[ResNet学习](https://blog.csdn.net/xxy0118/article/details/78324256)


### 4. MobileNet-V1 & MobileNet -V2

[V1论文地址](https://arxiv.org/abs/1704.04861)

[V2论文地址](https://arxiv.org/abs/1801.04381)

MobileNet是Google团队针对移动端提出的高效图像识别网络，深入的研究了Depthwise Separable Convolutions使用方法后设计出MobileNet，Depthwise Separable Convolutions的本质是冗余信息更少的稀疏化表达。在此基础上给出了高效模型设计的两个选择：宽度因子(Width Multiplier)和分辨率因子(Resolution Multiplier)；通过权衡大小、延迟时间以及精度，来构建规模更小、速度更快的MobileNet。

MobileNet V2是之前MobileNet V1的改进版。MobileNet V1中主要是引入了Depthwise Separable Convolution代替传统的卷积操作，相当于实现了spatial和channel之间的解耦，达到模型加速的目的，整体网络结构还是延续了VGG网络直上直下的特点。和MobileNet V1相比，MobileNet V2主要的改进有两点：1、Linear Bottlenecks。也就是去掉了小维度输出层后面的非线性激活层，目的是为了保证模型的表达能力。2、Inverted Residual block。该结构和传统residual block中维度先缩减再扩增正好相反，因此shotcut也就变成了连接的是维度缩减后的feature map。

[深度解读谷歌MobileNet](https://blog.csdn.net/t800ghb/article/details/78879612)

[轻量化网络：MobileNet-V2](https://blog.csdn.net/u011995719/article/details/79135818)

### 5. U-NET

[论文地址](http://arxiv.org/abs/1505.04597)

[项目地址](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

这是Encoder-Decoder网络的一种，在无监督学习中的框架，利用conv与deconv降维升维来进行学习，分别叫做encoder与decoder编码解码，一般基于卷积网络，encoder后相当于学习到了特征，而decoder后相当于还原了图像，既可以用输入图像进行训练，训练好一层加深一层。再可以利用有监督微调，从而达到分类或者图像转换的目的。

### 6. GAN

总结：利用两个网络对抗生成模型，生成器与辨别器，生成器输入图像，生成所需图像，辨别器辨别所需图像与生成图像，使生成器的生成图像骗过辨别器。


### 7.DenseNet

CVPR17 的Best Paper，模型体积小，准确率高。我们利用densenet + ctc进行进行OCR文字识别训练，效果不错。比基于resnet、vgg等基础网络的ocr效果好。
