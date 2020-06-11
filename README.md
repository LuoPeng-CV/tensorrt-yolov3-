#一、Python版
###使用的是TensorRT 7.0官方python用例，主要包括一下几个过程
- 1.将Darknet得到的cfg和weights文件转换成onnx模型
- 2.使用onnx模型生成.trt文件并对图片进行检测
- 3.切换FP16

##1.Darknet-->ONNX
**python yolov3_to_onnx.py**
  首先得安装onnx，pip安装即可,然后修改py文件中的一些参数，包括cfg文件、weights文件的路径，以及输出向量的大小等：
![修改py文件中的一些参数](https://upload-images.jianshu.io/upload_images/5955013-b7ab7d3c75858a6d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
运行后报错：
![运行报错](https://upload-images.jianshu.io/upload_images/5955013-21c1ff26afc2619f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
主要原因是TensorRT版本和ONNX版本匹配问题，经多次试验得出结果：
###***TensorRT 5.1.5与ONNX 1.4.1相匹配，TensorRT 7.0.0与ONNX 1.7.0相匹配***
使用其他版本会报错。
![无错版](https://upload-images.jianshu.io/upload_images/5955013-1bcd8b7b91418392.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
若不报错将会生成对应的ONNX模型。
##2.ONNX --> TensorRT
**python3 onnx_to_tensorrt.py**
修改参数：包括onnx路径，要生成的trt引擎路径，输出向量大小，模型608-->416等等：
![onnx_to_tensorrt](https://upload-images.jianshu.io/upload_images/5955013-5aa209de24b99661.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
运行可能报错：

![报错1](https://upload-images.jianshu.io/upload_images/5955013-dd5fcb86d3f8fa92.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

` sudo apt-get -y --force-yes install python-pycuda` 

报错2：
```Reading engine from file yolov3-bun-fp32.trt
[TensorRT] WARNING: TensorRT was linked against cuDNN 7.6.3 but loaded cuDNN 7.5.0
[TensorRT] WARNING: TensorRT was linked against cuDNN 7.6.3 but loaded cuDNN 7.5.0
[TensorRT] WARNING: Current optimization profile is: 0. Please ensure there are no enqueued operations pending in this context prior to switching profiles
Running inference on image /media/luopeng/F/TensorRT-7.0.0.11/samples/python/yolov3_onnx/images/nu2.jpg...
Traceback (most recent call last):
  File "onnx_to_tensorrt.py", line 188, in <module>
    main()
  File "onnx_to_tensorrt.py", line 168, in main
    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
  File "onnx_to_tensorrt.py", line 168, in <listcomp>
    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
ValueError: cannot reshape array of size 7581 into shape (1,21,13,13)
```
修改get_engine中模型输入大小和data_processing.py中的类别数量：
![类别数量](https://upload-images.jianshu.io/upload_images/5955013-8986be25e7de52af.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![网络输入大小](https://upload-images.jianshu.io/upload_images/5955013-913a62b9cfa8fc94.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##3. 切换FP16
设置`builder.fp16_mode = fp16_on`
![切换FP16](https://upload-images.jianshu.io/upload_images/5955013-1fc6446e571348f9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

检测结果：
![检测结果](https://upload-images.jianshu.io/upload_images/5955013-943084e57ea69ea9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


#二、C++版
> [https://github.com/wang-xinyu/tensorrtx/tree/master/yolov3](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov3)

使用的是第三方实现的C++实现用例，主要包括以下过程：
- 1.将Darknet得到的cfg和weights文件转成yolov3.wts（pytorch版yolov3的权重文件)。
- 2.编译C++版TensorRT for YOLOv3
- 3.生成yolov3.engine用于推理加速并对图片进行检测
- 4.切换FP16
```
环境：
TensorRT 7.0.0
CUDA10.0
OpenCV with contrib 3.4.5
```
##1. 生成yolov3.wts
导入pytorch版的yolov3，将darknet上训练好的yolov3.weights放到该工程，使用gen_wts.py生成yolov3.wts
```
git clone https://github.com/wang-xinyu/tensorrtx.git
git clone https://github.com/ultralytics/yolov3.git
// download its weights 'yolov3.pt' or 'yolov3.weights'
cd yolov3
cp ../tensorrtx/yolov3/gen_wts.py .
python gen_wts.py yolov3.weights
// a file 'yolov3.wts' will be generated.
```

## 2.编译C++版TensorRT for YOLOv3
将生成的yolov3.wts放入tensorrtx/yolov3目录，在yolov3.cpp中可修改yolov3.wts路径，NMS thresh，BBox confidence thresh，使用FP16还是FP32等；在yolov3.h中可修改网络输入大小、类别数量等。
开始编译：
```
mkdir build
cd build
cmake ..
make
```
在编译过程中可能出现的问题：
```
1）
fatal error: NvInfer.h: No such file or directory
#include "NvInfer.h"
^~~~~~~~~~~
compilation terminated.
CMake Error at myplugins_generated_mish.cu.o.Debug.cmake:219 (message):
Error generating
/media/tensorrtx/yolov4/build/CMakeFiles/myplugins.dir//./myplugins_generated_mish.cu.o

CMakeFiles/myplugins.dir/build.make:70: recipe for target 'CMakeFiles/myplugins.dir/myplugins_generated_mish.cu.o' failed
make[2]: *** [CMakeFiles/myplugins.dir/myplugins_generated_mish.cu.o] Error 1
CMakeFiles/Makefile2:72: recipe for target 'CMakeFiles/myplugins.dir/all' failed
make[1]: *** [CMakeFiles/myplugins.dir/all] Error 2
Makefile:83: recipe for target 'all' failed
make: *** [all] Error 2
```
原因是TensorRT的头文件没有被找到，解决方法是将TensorRTx.x.x/include加入环境变量或将这些头文件复制到/usr/include下。
```
2）能编译通过但在执行的时候报错：
段错误(核心已转储)
```

![段错误(核心已转储)](https://upload-images.jianshu.io/upload_images/5955013-506afdfe7748b30f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
主要原因是opencv编译的问题，带contrib的opencv编译通过后错误消失。

##3.生成yolov3.engine用于推理加速并对图片进行检测
编译完成后

```
sudo ./yolov3 -s             // serialize model to plan file i.e. 'yolov3.engine'
sudo ./yolov3 -d  ../images/      //deserialize plan file and run inference, the images will be processed.
```
-s：生成推理引擎文件, -d：加载引擎文件开始推理
结果：


![_n008-2018-07-27-12-07-38-0400__CAM_BACK__1532708580537558.jpg](https://upload-images.jianshu.io/upload_images/5955013-902120dfd7b89ed6.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 4. 切换FP16和FP32
在yolov3.cpp中加入
```
#ifdef USE_FP16
    std::cout << "using FP16" <<std::endl;
    config->setFlag(BuilderFlag::kFP16);
#endif
```
FP32与FP16速度对比：
FP32推理一张图片约15ms，FP16约6ms
![推理时间](https://upload-images.jianshu.io/upload_images/5955013-2dd81934dd2313f8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)





