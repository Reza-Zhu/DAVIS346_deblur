主线程main：相机及模型初始化,cv show
DAVIS 类， thread_davis线程：采集帧图像/事件，并做必要的预处理
RenderFrame类， deblur线程：去除hot_pixel，模型推理
具体细节参考注释
