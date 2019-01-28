# -*- coding:utf-8 -*-
# 用于模型的单张图像分类操作
import os
import sys
caffe_root='/home/sugon/caffe_workspace/SSD/caffe/'#指定根目录
sys.path.insert(0,caffe_root+'python')
os.environ['GLOG_minloglevel'] = '2' # 将caffe的输出log信息不显示，必须放到import caffe前
import caffe # caffe 模块
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

#CPU或GPU模型转换
caffe.set_mode_cpu()
#caffe.set_device(0)
#caffe.set_mode_gpu()

# 网络参数（权重）文件
caffemodel = caffe_root + 'models/darknet53/darknet53_ilsvrc12_iter_800730.caffemodel'
# 网络实施结构配置文件
deploy = caffe_root + 'models/darknet53/darknet53_deploy.prototxt'
synset_words = caffe_root + 'data/ilsvrc12/synset_words.txt'
gt_label_file = caffe_root + 'data/ilsvrc12/val.txt'
k=''
img_root = caffe_root + 'test_images/'#保存测试图片的集合
filelist=[]
filenames=os.listdir(img_root)
for fn in filenames:
    fullfilename = os.path.join(img_root,fn)
    filelist.append(fullfilename)
# 网络实施分类
net = caffe.Net(deploy,  # 定义模型结构
                caffemodel,  # 包含了模型的训练权值
                caffe.TEST)  # 使用测试模式(不执行dropout)

# 图像预处理
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))# (h,w,c)--->(c,h,w)
transformer.set_mean('data', np.array([104,117,123])) #注意是 BGR
'''
或者通过caffe_root/build/tools/compute_image_mean 计算图像均值得到xxx.npy
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')# 加载ImageNet图像均值 (随着Caffe一起发布的)
mu = mu.mean(1).mean(1)  # 对所有像素值取平均以此获取BGR的均值像素值
transformer.set_mean('data', mu)
print 'mean-subtracted values:', zip('BGR', mu)
'''
transformer.set_raw_scale('data', 255)# rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))# RGB--->BGR
'''
可能你对均值设置成BGR有疑问，不是到后面才把RGB转为BGR吗?
其实transformer.setXXX()这些只是设置属性，实际执行顺序是参考附录preprocess函数定义
'''

# 分类单张图像img
def classification(img, net, transformer, synset_words,label_num):
    # JPEG=cv2.imread(img)
    im = caffe.io.load_image(img)
    # 导入输入图像
    net.blobs['data'].data[...] = transformer.preprocess('data', im)

    start = time.clock()
    # 执行测试
    net.forward()
    end = time.clock()
    # print 'classification time: %f s' % (end - start)

    # 查看结果
    labels = np.loadtxt(synset_words, str, delimiter='\t')
    category = net.blobs['prob'].data[0].argmax()    
    # 或者category = net.blobs['prob'].data[0].flatten().argsort()[-1]

    gt_class_names = labels[int(label_num)].split(',')
    gt_class_name = gt_class_names[0]

    test_class_names = labels[int(category)].split(',')
    test_class_name = test_class_names[0]
    
    print 'gt_class_name: ',gt_class_names
    print 'test_class_name: ',test_class_names
    
    cv2.putText(im, '  gt: '+gt_class_name, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (55, 255, 255), 2)
    cv2.putText(im, 'test: '+test_class_name, (0, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 155,55), 2)

    # 显示结果
    # cv2.imshow('JPEG', JPEG)
    # cv2.waitKey(0) # 无限期等待输入
    # if k==27: # 如果输入ESC退出
    #     cv2.destroyAllWindows() 
    # elif k==ord('s'): # 如果输入s,保存
    #     cv2.imwrite('test.png',img)
    #     print "SAVE OK!"
    #     cv2.destroyAllWindows()
    plt.imshow(im)
    plt.show()

# 处理图像
for i in range(0,len(filelist)):
    img_num = raw_input("Enter Img Number: ")
    if img_num == '': 
        break
    img = 'ILSVRC2012_val_'+ '{:0>8}'.format(img_num) + '.JPEG'
    # print img
    # 显示真实label
    label_num=''
    for line in open(gt_label_file):
        if img in line:
            label_num = line.split(' ')[-1]
    classification(img_root + img,net,transformer,synset_words,label_num)