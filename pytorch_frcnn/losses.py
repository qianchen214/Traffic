import torch
import torch.nn as nn
import torch.nn.functional as F

# if K.image_dim_ordering() == 'tf':#tensorflow或者theano
#     import tensorflow as tf

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0  # 1.0  TODO

lambda_cls_regr = 1.0
lambda_cls_class = 1.0  # 1.0

epsilon = 1e-4


def rpn_loss_regr(num_anchors):
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        flag = True
        if flag:
            x = y_true[:, 4 * num_anchors:, :, :] - y_pred
            x_abs = torch.abs(x)
            x_bool = torch.less_equal(x_abs, 1.0)
            return lambda_rpn_regr * torch.sum(
                y_true[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / torch.sum(
                epsilon + y_true[:, :4 * num_anchors, :, :])
        else:
            x = y_true[:, :, :, 4 * num_anchors:] - y_pred
            x_abs = torch.abs(x)
            x_bool = torch.FloatTensor(torch.less_equal(x_abs, 1.0))

            return lambda_rpn_regr * torch.sum(
                y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / torch.sum(
                epsilon + y_true[:, :, :, :4 * num_anchors])

    return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        flag = False
        if flag:
            return lambda_rpn_class * torch.sum(y_true[:, :, :, :num_anchors] * F.binary_cross_entropy(y_pred[:, :, :, :],
                                                                                                  y_true[:, :, :,
                                                                                                  num_anchors:])) / torch.sum(
                epsilon + y_true[:, :, :, :num_anchors])
        else:
            return lambda_rpn_class * torch.sum(y_true[:, :num_anchors, :, :] * F.binary_cross_entropy(y_pred[:, :, :, :],
                                                                                                  y_true[:,
                                                                                                  num_anchors:, :,
                                                                                                  :])) / torch.sum(
                epsilon + y_true[:, :num_anchors, :, :])

    return rpn_loss_cls_fixed_num#不知道选哪个，需要测试


def class_loss_regr(num_classes):
    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4 * num_classes:] - y_pred
        x_abs = torch.abs(x)
        x_bool = torch.FloatTensor(torch.less_equal(x_abs, 1.0))
        return lambda_cls_regr * torch.sum(
            y_true[:, :, :4 * num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / torch.sum(
            epsilon + y_true[:, :, :4 * num_classes])

    return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    crossentropyloss = nn.CrossEntropyLoss()
    return lambda_cls_class*torch.mean(crossentropyloss(y_true[0, :, :], y_pred[0, :, :]))#mean 求均值 cross交叉熵
