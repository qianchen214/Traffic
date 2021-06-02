# -*- coding: utf-8 -*-
"""MobileNet model for Keras.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
# import RoiPoolingConv

from pytorch_frcnn.RoiPoolingConv import RoiPoolingConv
import torch.nn as nn
import torch

from torchvision import models


WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_1_0_224_tf_no_top.h5'
alpha = 1
depth_multiplier = 1
dropout = 1e-3
include_top = True
  # todo : 看需要删除哪些对象

# def ():#?Nobody Use
#     if K.image_dim_ordering() == 'th':
#         print('pretrained weights not available for Mobilenet with theano backend')
#         return
#     else:
#         return WEIGHT_PATH

def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length // 16

    return get_output_length(width), get_output_length(height)

class MobileNet(nn.Module):
    def __init__(self,num_anchors,input_rois,num_rois,nb_classes=21):
        super(MobileNet, self).__init__()
        self.num_anchors = num_anchors
        self.input_rois = input_rois
        self.num_rois = num_rois
        self.nb_classes = nb_classes
        self.input_shape = 512
        self.shape = 512

        def conv_bn(input,output,stride):
            return nn.Sequential(
                nn.Conv2d(input,output,3,stride,1,bias=False),
                nn.BatchNorm2d(output),
                nn.ReLU6(inplace=True))
        def conv_dw(input,output,stride):
            return nn.Sequential(
                nn.Conv2d(input,input,3,stride,1,groups=input,bias=False),
                nn.BatchNorm2d(input),
                nn.ReLU6(inplace=True),

                nn.Conv2d(input,output,1,1,0,bias=False),
                nn.BatchNorm2d(output),
                nn.ReLU6(inplace=True)
            )

        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.model =  nn.Sequential(*list(self.mobilenet.features)[:-4])
        
        self.model[14] = conv_dw(96, 512, 1)
        #print(self.model)
        

        # self.model = nn.Sequential(
        #     conv_bn(3, 32, 2),
        #     conv_dw(32, 64, 1),
        #     conv_dw(64, 128, 2),
        #     conv_dw(128, 128, 1),
        #     conv_dw(128, 256, 2),
        #     conv_dw(256, 256, 1),
        #     conv_dw(256, 512, 2),

        #     conv_dw(512, 512, 1),
        #     conv_dw(512, 512, 1),
        #     conv_dw(512, 512, 1),
        #     conv_dw(512, 512, 1),
        #     conv_dw(512, 512, 1),

        # #     # conv_dw(512, 1024, 2),
        # #     # conv_dw(1024, 1024, 1),
        # #     # nn.AvgPool2d(7),
        # )
        # #self.model = models.resnet50(pretrained=True)

        self.rpn = nn.Sequential(
            nn.Conv2d(512, 256, (3, 3), padding=1),
            nn.ReLU6()
        )

        self.rpn_class = nn.Sequential(
            nn.Conv2d(256,2*self.num_anchors,(1,1)),
            nn.Sigmoid()
        )
        self.rpn_reg = nn.Sequential(
            nn.Conv2d(256, self.num_anchors*4*2, (1, 1)),
            #nn.Linear(self.num_anchors*4,self.num_anchors*4)
        )

        self.classifier = nn.Sequential(
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
            nn.Dropout(dropout),
            nn.Flatten()
            # nn.Linear(self.shape,self.nb_classes),
            # nn.Linear(self.shape,4*(self.nb_classes-1)),
        )
        self.classifier_class = nn.Sequential(
            nn.Linear(1024,self.nb_classes),
            nn.Softmax()
        )
        self.classifier_regr = nn.Sequential(
            nn.Linear(1024,8*(self.nb_classes - 1))
        )



        self.fc = nn.Linear(1024, 1000)

    def forward(self,x,type="all"):
        if type == "rpn":
            #print(x.size())
            x = self.model(x)
            # x = self.model.conv1(x)
            # x = self.model.bn1(x)
            # x = self.model.relu(x)
            # x = self.model.maxpool(x)
            # x = self.model.layer1(x)
            # x = self.model.layer2(x)
            self.shape = x.shape
            x_middle = self.rpn(x)
            #print(x_middle)
            x_class = self.rpn_class(x_middle)
            x_reg = self.rpn_reg(x_middle)
            return [x_class,x_reg,x]
        elif type =="classifier":
            pooling_regions = 14  # todo: 使用更小的pooling_regions ? 这样会更快吗？
            input_shape = (self.num_rois, 512, 7, 7)
            self.input_shape = input_shape

            middle = RoiPoolingConv(pooling_regions, self.num_rois)
            middle.build(self.input_shape)
            out_roi_pool = middle.call(x)
            output = self.classifier(out_roi_pool[0])

            out_class = self.classifier_class(output)
            out_regr = self.classifier_regr(output)
            #print(8*(self.nb_classes-1),"aaa")
            return [out_class,out_regr]
        else:
            self.shape = x.shape
            x_middle = self.rpn(x)
            x_class = self.rpn_class(x_middle)
            x_reg = self.rpn_reg(x_middle)

            pooling_regions = 14  # todo: 使用更小的pooling_regions ? 这样会更快吗？
            input_shape = (self.num_rois, 512, 7, 7)


            middle = RoiPoolingConv(pooling_regions, self.num_rois)([x,self.input_rois])
            middle.build(self.input_rois)
            out_roi_pool = middle.call(x)
            out = self.classifier(out_roi_pool[0])
            self.shape = out.shape

            m = nn.Linear(self.shape, self.nb_classes)
            out_class = m(out)
            m = nn.Linear(self.shape, 4 * (self.nb_classes) - 1)
            out_regr = m(out)
            return [x_class,x_reg,x,out_class,out_regr]

        # x = x.view(-1,1024)
        # x = self.fc(x)


def mobileNet(num_anchors,input_rois,num_rois,nb_classes=21):
    model = MobileNet(num_anchors,input_rois,num_rois,nb_classes).cuda()
    return model


def rpn(base_layers, num_anchors):  # 9 anchors are used here
    m = nn.Conv2d(base_layers.shape,256,(3,3),padding=1)
    x = m(base_layers)
    x = nn.ReLU6(x)

    m = nn.Conv2d(x.shape,num_anchors,(1,1),padding=1)
    x_class = m(x)
    x_class = torch.sigmoid(x_class)

    m = nn.Conv2d(x.shape,num_anchors*4,(1,1))
    x_regr = m(x)
    m = nn.Linear(x_regr.shape,x_regr.shape)
    x_regr = m(x_regr)

    return [x_class, x_regr, base_layers]


def classifier(model,base_layers, input_rois, num_rois, nb_classes=21, trainable=True):  # background and interesting objects
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

    pooling_regions = 14   # todo: 使用更小的pooling_regions ? 这样会更快吗？
    input_shape = (num_rois, 14, 14, int(512 * alpha))
    # pooling_regions = 7
    # input_shape = (num_rois, 512, 7, 7)

    middle = RoiPoolingConv(pooling_regions, num_rois)
    middle.build(input_rois)
    out_roi_pool = middle.call(base_layers)


    #out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=trainable)
    out = model.classifier_layers(out_roi_pool,input_shape)
    out = nn.Flatten(out)

    m = nn.Linear(out.shape,nb_classes)
    out_class = m(out)

    m = nn.Linear(out.shape,4*(nb_classes-1))
    out_regr = m(out)
    return [out_class, out_regr]




