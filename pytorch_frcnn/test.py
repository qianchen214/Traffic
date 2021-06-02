
import time
import torch

import numpy as np

# def speed(model, name):
#     t0 = time.time()
#     input = torch.rand(1, 3, 224, 224).cuda()
#
#     t1 = time.time()
#
#     output_rpn = model(input,"rpn")
#     output_classifier = model(input,"classifier")
#
#     #output = model.classifier_layers(output,512)
#     print(output)


roi_input = (None, 4)

num_anchors = 9

input_shape = (4, 512, 7, 7)
print(input_shape)
# # define the base network (resnet here, can be VGG, Inception, etc)
# model_rpn = mobilenet.MobileNet(num_anchors,roi_input, 9)

# model = mobilenet.mobileNet()
# speed(model,"mobilenet")
a = [np.zeros((224,224,3)), np.zeros((224,224,3)), np.zeros((224,224))]
list_temp = np.array(a)
print(list_temp.shape)











