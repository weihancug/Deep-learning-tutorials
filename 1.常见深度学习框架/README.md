### 常用深度学习框架



深度学习研究的热潮持续高涨，各种开源深度学习框架也层出不穷，其中包括TensorFlow、Caffe、Keras、CNTK、Torch7、MXNet、Leaf、Theano、DeepLearning4、Lasagne、Neon等。

| 框架 | 机构 | 支持语言 | Stars | Forks |
| --------------------------------------------------------- | ----------- | ------------------- | ---------- | ----- |
| [Caffe](https://github.com/BVLC/caffe)                    | BVLC        | C++/Python/Matlab   |    ![](https://img.shields.io/github/stars/BVLC/caffe.svg)   | ![](https://img.shields.io/github/forks/BVLC/caffe.svg) |
| [CNTK](https://github.com/Microsoft/CNTK)                 | Microsoft   | C++                 |    ![](https://img.shields.io/github/stars/Microsoft/CNTK.svg)   | ![](https://img.shields.io/github/forks/Microsoft/CNTK.svg) |
| [Keras](https://github.com/keras-team/keras)                     | François Chollet    | Python              |    ![](https://img.shields.io/github/stars/keras-team/Keras.svg)   | ![](https://img.shields.io/github/forks/keras-team/Keras.svg) |
| [Tensorflow](https://github.com/tensorflow/tensorflow)    | Google      | Python/C++/Go...    |    ![](https://img.shields.io/github/stars/tensorflow/Tensorflow.svg)   | ![](https://img.shields.io/github/forks/tensorflow/Tensorflow.svg) |
| [MXNet](https://github.com/apache/incubator-mxnet)        | DMLC        | Python/C++/R...     |    ![](https://img.shields.io/github/stars/apache/incubator-mxnet.svg)   | ![](https://img.shields.io/github/forks/apache/incubator-mxnet.svg) |
| [PyTorch](https://github.com/pytorch/pytorch)             | Facebook    | Python              |    ![](https://img.shields.io/github/stars/pytorch/pytorch.svg)   | ![](https://img.shields.io/github/forks/pytorch/pytorch.svg) |




### 推荐框架


目前众多的深度学习框架，使用者只要选择适合自己的框架即可，我们在日常使用中，考虑到训练的快捷程度，部署难度以及对CNN、RNN模型的直接程度，推荐以下几款深度学习框架。

#### 1.Keras

Keras 提供了简单易用的 API 接口，入门快，特别适合初学者入门。其后端采用 TensorFlow, CNTK，以及 Theano。另外，Deeplearning4j 的 Python 也是基于 Keras 实现的。Keras 几乎已经成了 Python 神经网络的接口标准。

#### 2.TensorFlow

谷歌出品，追随者众多。代码质量高，支持模型丰富，支持语言多样， TensorBoard 可视化工具使用方便。

#### 3.PyTorch

很简洁、易于使用、支持动态计算图而且内存使用很高效，因此越来越受欢迎。


#### 4.总结

其实，最好在PyTorch和TensorFlow二选一，一个好的框架应该要具备三点：
1.对大的计算图能方便的实现；
2.能自动求变量的导数；
3.能简单的运行在GPU上。
pytorch都做到了，但是现在很多公司用的都是TensorFlow，而pytorch由于比较灵活，在学术科研上用得比较多一点。TensorFlow在GPU的分布式计算上更为出色，在数据量巨大时效率比pytorch要高一些。
具体需求根据自己理解选择把。学术相对推荐pytorch。





### 参考资料
1. [香港浸会大学深度学习框架Benchmark](http://dlbench.comp.hkbu.edu.hk/?v=v8)
2. [DeepLearningFrameworks](https://github.com/ilkarman/DeepLearningFrameworks)
3. [博客](http://app.myzaker.com/news/article.php?pk=5a13b55c1bc8e05d71000016)
4. [开发者如何选择深度学习框架?](https://www.zhihu.com/question/68114194/answer/465874315)
