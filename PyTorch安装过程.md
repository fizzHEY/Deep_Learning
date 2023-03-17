### 基于anconda虚拟环境的PyTorch安装过程

#### 1、CUDA、cuDNN下载

##### 确保自己显卡驱动的版本大于等于下载的cuda版本要求

![image-20221022094721535](C:\Users\zls05\AppData\Roaming\Typora\typora-user-images\image-20221022094721535.png)

如图最高支持CUDA11.8版本

+ CUDA官网下载[CUDA Toolkit Archive | NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit-archive)

![img](https://img-blog.csdnimg.cn/85a5a1d7b9db40ddad12fe076f93c200.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQmlsbGll5L2_5Yqy5a2m,size_20,color_FFFFFF,t_70,g_se,x_16)

双击安装，默认路径即可

![img](https://img-blog.csdnimg.cn/d2af91db838e4cafad252b7281ab81ce.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQmlsbGll5L2_5Yqy5a2m,size_19,color_FFFFFF,t_70,g_se,x_16)

![img](https://img-blog.csdnimg.cn/4988f107977c454fa96550347e239862.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQmlsbGll5L2_5Yqy5a2m,size_19,color_FFFFFF,t_70,g_se,x_16)记住安装路径，后续cuDNN需要找到这里

pytorch版本对应，具体参照官网

![img](https://img-blog.csdnimg.cn/img_convert/3e03b77f849bb0db1e3b66b198608965.png)

##### 查看环境变量

点击设置-->搜索高级系统设置-->查看环境变量

![image-20221022095702515](C:\Users\zls05\AppData\Roaming\Typora\typora-user-images\image-20221022095702515.png)

##### 验证是否安装成功：

运行cmd，输入nvcc --version 即可查看版本号；

![image-20221022095843191](C:\Users\zls05\AppData\Roaming\Typora\typora-user-images\image-20221022095843191.png)

##### cuDNN下载

+ cuDNN下载网址https://developer.nvidia.com/rdp/cudnn-download

先注册，才能下载，选择与自己cuda版本对应的cudnn版本。

下载后将压缩包解压操作，

![img](https://img-blog.csdnimg.cn/ccabf380ed4843b2aa3401b3bf8a765e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQmlsbGll5L2_5Yqy5a2m,size_16,color_FFFFFF,t_70,g_se,x_16)

将三个文件夹复制到刚刚cuda的安装路径,默认路径如下：

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1
```

添加如下系统变量到path

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\libnvvp
```

![image-20221022101155796](C:\Users\zls05\AppData\Roaming\Typora\typora-user-images\image-20221022101155796.png)

配置完成后，验证是否配置成功，主要使用CUDA内置的deviceQuery.exe和bandwidthTest.exe：在安装目录下的 …\extras\demo_suite打开dos栏, 然后分别执行bandwidthTest.exe和deviceQuery.exe（进到目录后需要直接输“bandwidthTest.exe”和“deviceQuery.exe”）,应该得到下图: 出现pass说明安装成功

![image-20221022101427579](C:\Users\zls05\AppData\Roaming\Typora\typora-user-images\image-20221022101427579.png)

#### 2、pytorch环境配置

1. 打开anconda安装目录下的Scripts文件夹，地址栏输入cmd打开dos栏，输入如下命令：

   `conda create -n pyTorch python=3.7 `，点击回车。其中`pyTorch`是环境的名字，自己定义也可以。`python=3.7`是这个环境将使用3.7的python版本。

![image-20221022102358858](C:\Users\zls05\AppData\Roaming\Typora\typora-user-images\image-20221022102358858.png)

![image-20221022103654401](C:\Users\zls05\AppData\Roaming\Typora\typora-user-images\image-20221022103654401.png)

输入y，回车确定，等待下载结束。

![image-20221022103950066](C:\Users\zls05\AppData\Roaming\Typora\typora-user-images\image-20221022103950066.png)

2. 打开刚刚安装好的环境

![image-20221022105455635](C:\Users\zls05\AppData\Roaming\Typora\typora-user-images\image-20221022105455635.png)

​		同样使用 `dos` 进入 `pyTorchEnv` 环境的 `Scripts`文件夹，然后激活`pyTorch`。方法同上边，找到 `pyTorch` 中的 `Scripts` 文件夹，在路径栏输入 `cmd` 回车进入`dos`。然后使用 `activate pyTorch` 激活它。

打开pytorch官网[PyTorch](https://pytorch.org/)

选择自己的配置

![image-20221022105814510](C:\Users\zls05\AppData\Roaming\Typora\typora-user-images\image-20221022105814510.png)

将复制的命令`conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge`放入刚才打开的`dos`窗口，回车进行安装

输入y，回车确认。等待下载完成，出现done不报错，安装成功。

输入python

![image-20221022111136862](C:\Users\zls05\AppData\Roaming\Typora\typora-user-images\image-20221022111136862.png)

导入torch包，不报错就是成功。

![image-20221022111222913](C:\Users\zls05\AppData\Roaming\Typora\typora-user-images\image-20221022111222913.png)

#### 3、pycharm导入pytorch环境

1. 打开项目设置，解释器选项

![image-20221022111423441](C:\Users\zls05\AppData\Roaming\Typora\typora-user-images\image-20221022111423441.png)

2. 添加解释器

![image-20221022111633753](C:\Users\zls05\AppData\Roaming\Typora\typora-user-images\image-20221022111633753.png)

3. 选择刚刚安装好的pytorch目录下的python.exe

![image-20221022111648921](C:\Users\zls05\AppData\Roaming\Typora\typora-user-images\image-20221022111648921.png)

4. 点击确定，等待部署完成