# coding:utf-8
from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time

import torch
import torch.optim as optim
import torch.nn as nn

from pytorch_frcnn import roi_helpers

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use GTX 1080

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# def get_session(gpu_fraction=0.3):
#     """
#     This function is to allocate GPU memory a specific fraction
#     Assume that you have 6GB of GPU memory and want to allocate ~2GB
#     """

#     num_threads = os.environ.get('OMP_NUM_THREADS')
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

#     if num_threads:
#         return tf.Session(config=tf.ConfigProto(
#             gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
#     else:
#         return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# K.set_session(get_session(0.3))  # using about 20% ~ ３0% of total GPU Memory　ｉs sufficient!

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", dest="num_rois",
                  help="Number of ROIs per iteration. Higher means more memory use.", default=512) #32
# 测试的时候roi取多一点，比如2000个
parser.add_option("--config_filename", dest="config_filename", help=
"Location to read the metadata related to the training (generated when training).",
                  default="config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.",
                  default='resnet50')

(options, args) = parser.parse_args()

if not options.test_path:  # if filename is not given
    parser.error('Error: path to test data must be specified. Pass --path to command line')

config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

if C.network == 'resnet50':
    import pytorch_frcnn.resnet as frcnn
elif C.network == 'vgg':
    import pytorch_frcnn.vgg as frcnn
elif C.network == 'mobilnet':
    import pytorch_frcnn.mobilenet as frcnn
    print('*********************** mobilenet *********************')

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.test_path


def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)


class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}  # 每种class用不同的框框出来
C.num_rois = int(options.num_rois)

if C.network == 'resnet50':
    num_features = 1024

elif C.network == 'vgg':
    num_features = 512

elif C.network == 'mobilnet':
    num_features = 512

# input_shape_img = (None, None, 3)
# input_shape_features = (None, None, num_features)

# img_input = torch.tensor(input_shape_img)
roi_input = torch.tensor((C.num_rois, 4))
# feature_map_input = torch.tensor(input_shape_features)

num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)

# define the base network (resnet here, can be VGG, Inception, etc)
model = frcnn.MobileNet(num_anchors,roi_input, C.num_rois, nb_classes=len(class_mapping))
model = model.to(device)

print('Loading weights from {}'.format(C.model_path))

weights_dict = torch.load(C.model_path)
model.load_state_dict(weights_dict)

mseloss = nn.MSELoss()

optimizer=optim.SGD(model.parameters(),lr=0.1)
optimizer.zero_grad()

# model_rpn.compile(optimizer='sgd', loss='mse')
# model_classifier.compile(optimizer='sgd', loss='mse')
# ########### plot model #################
# from keras.utils import plot_model
# plot_model(model_rpn, to_file='model_rpn1.png', show_shapes=True)
# plot_model(model_classifier, to_file='model_classifier1.png', show_shapes=True)

all_imgs = []

classes = {}

tick = []

bbox_threshold = 0.995  # 0.8

visualise = True

for idx, img_name in enumerate(sorted(os.listdir(img_path))):   # 读取文件夹中的所有图片
    if not img_name.lower().endswith(('.bmp', '.ppm', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    print("-----------------------------")
    print(img_name)
    st = time.time()
    filepath = os.path.join(img_path, img_name)

    img = cv2.imread(filepath)

    X, ratio = format_img(img, C)

    #X = np.transpose(X, (0, 2, 3, 1))
    X = torch.tensor(X).cuda()

    model.eval()
    [Y1, Y2, F] = model(X,"rpn")
    # get the feature maps and output from the RPN
    #[Y1, Y2, F] = model_rpn.predict(X)

    Y1 = Y1.cpu().detach().numpy()
    Y2 = Y2.cpu().detach().numpy()

    R = roi_helpers.rpn_to_roi(Y1, Y2, C, max_boxes=64, overlap_thresh=0.7 #
                               )  # RPN层bbox极大值进行抑制的阈值 todo
    # R: RPN层上的坐标
    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions  根据RPN得到roi
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0] // C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // C.num_rois:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model([F, ROIs],"classifier")
        P_cls = P_cls.unsqueeze(0).cpu().detach().numpy()

        #[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        for ii in range(P_cls.shape[1]):

            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= C.classifier_regr_std[0]  # todo 配置文件中的标准差
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)

            except:
                pass
            bboxes[cls_name].append(
                [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []

    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.3)
        # todo : 0.3 is good
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk, :]

            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

            cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                          (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 2)

            textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))
            all_dets.append((key, 100 * new_probs[jk]))


    print('Elapsed time = {}'.format(time.time() - st))
    tick.append(time.time() - st)
    print(all_dets)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.imwrite('./result/{}.png'.format(idx), img)

print(sum(tick))
