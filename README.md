## keras 计算机视觉例程及深入

PS：项目中用到了 datasets 和 models 等辅助资料不在仓库中备份。

### 环境

代码运行环境为 python 3.7.4，tensorflow 1.15.0，keras 2.2.4，笔记本电脑 Win10 系统，pyCharm IDE。

### 目录

本文记录使用 keras 运行与计算机视觉相关的例程。例程来自于 keras [官网](https://keras.io/examples/)。

- [Image classification from scratch](#Image classification from scratch)



### Image classification from scratch

#### 准备数据集

首先，下载好 Kaggle 猫狗分类数据集（786M）。

#### 去除损坏的图像

过滤掉头部没有字符串“JFIF”的编码错误的图像。

#### 生成数据集

#### 查看数据

#### 数据增强

这有助于将模型暴露于训练数据的不同方面，同时减缓过度拟合的速度。

#### 预处理

图像尺寸为 180 x 180，RGB 的数值在[0,255]范围，这对于神经网络来说并不理想；一般来说，应该使输入值变小，将值标准化为[0，1]。

这里有两种预处理方式，区别在于是在 GPU 上运行预处理，还是在 CPU 上运行预处理。

-  **Make it part of the model**

  使用此选项，您的数据扩充将在设备上进行，与模型执行的其余部分同步，这意味着它将受益于GPU加速。
  请注意，数据扩充在测试时是不活动的，因此输入样本将只在 `fit()` 期间进行扩充，而不是在调用`evaluate()` 或`predict()`时。
  如果你在 GPU 上训练，这是更好的选择。

- **apply it to the dataset**

  使用这个选项，您的数据扩充将在CPU上异步进行，并在进入模型之前进行缓冲。
  如果您正在进行CPU培训，这是一个更好的选择，因为它使数据扩充异步且无阻塞。

本人使用的笔记本电脑，因此选择了第二种方式。

#### 构建模型



#### 训练模型

