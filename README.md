### OICR-pytorch

* [论文原址](http://xxx.itp.ac.cn/abs/1704.00138) 
* 训练细节：
    * 去除倒数第二个池化层，并将其后的卷积层改为dilated conv
    * vgg16中没有的层bias初始化为0，weight根据标准差0.01，均值0的高斯分布初始化
    * bs=4(即每四个backward进行一次step), 
    * 不对bias做权重衰减，且bias的梯度乘于2，优化器为SGD(momentum=0.9)
    * 不对ssw生成的roi做操作(包括去除过小的roi和去除重复的roi)，**做操作会影响模型的表现**
        * 此为实验的推测，当去除过小roi和重复roi后，模型最终表现只有20附近的mAP表现
        * 下面给出的是仅仅去除过小roi和不处理roi的实验结果(为了加快训练，此处对img的放缩仅放缩至较小scale)
* 模型表现
* Model_1 在scale=(480, )情况下，去除过小roi的设置下的模型表现(Lr=1e-4, Epoch：23)：
![model1.png]('img/model1.png')
* Model_2 在scale=(480, 576, 688)情况下，去除过小roi的设置下的模型表现(Lr=1e-4, Epoch：26)：
![model2.png]('img/model2.png')