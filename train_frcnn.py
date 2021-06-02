from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle

import torch
import torch.optim as optim
import pytorch_frcnn.mobilenet as frcnn
from pytorch_frcnn import config, data_generators
from pytorch_frcnn import losses as losses
import pytorch_frcnn.roi_helpers as roi_helpers

import torch
import torch.optim as optim
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
                default="pascal_voc")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=32)
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
                  action="store_true", default=False)
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=2000)
parser.add_option("--config_filename", dest="config_filename", help=
                "Location to store all the metadata related to the training (to be used when testing).",
                default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='./model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")

(options, args) = parser.parse_args()

if not options.train_path:   # if filename is not given
    parser.error('Error: path to training data must be specified. Pass --path to command line')

if options.parser == 'pascal_voc':
    from pytorch_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
    from pytorch_frcnn.simple_parser import get_data
else:
    raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

# pass the settings from the command line, and persist them in the config object
C = config.Config()

C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

C.model_path = options.output_weight_path
C.num_rois = int(options.num_rois)

if options.network == 'vgg':
    C.network = 'vgg'
    from pytorch_frcnn import vgg as frcnn
elif options.network == 'resnet50':
    from pytorch_frcnn import resnet as frcnn
    C.network = 'resnet50'
elif options.network == 'mobilenet':
    C.network = 'mobilnet'
    from pytorch_frcnn import mobilenet as frcnn
    print('*********************** mobilenet *********************')
else:
    print('Not a valid model')
    raise ValueError

# check if weight path was passed via command line
# if options.input_weight_path:
#     C.base_net_weights = options.input_weight_path
# else:
#     # set the path to weights based on backend and model
#     C.base_net_weights = frcnn.get_weight_path()

all_imgs, classes_count, class_mapping = get_data(options.train_path)

if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C,config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, frcnn.get_img_output_length, mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, frcnn.get_img_output_length,mode='val')

roi_input = (None, 4)

num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)

# define the base network (resnet here, can be VGG, Inception, etc)
model = frcnn.MobileNet(num_anchors,roi_input, C.num_rois, nb_classes=len(class_mapping))
model = model.to(device)

# try:
#     print('loading weights from {}'.format(C.base_net_weights))
#     weights_dict = torch.load(C.base_net_weights)
#     model.load_state_dict(weights_dict)
#     print(model)
#     #model_classifier.load_state_dict(weights_dict)

# except:
#     print('Could not load pretrained model weights. Weights can be found in the keras application folder \
#         https://github.com/fchollet/keras/tree/master/keras/applications')


mseloss = nn.MSELoss()

optimizer=optim.Adam(model.parameters(),lr=1e-5)
# optimizer_classifier=optim.Adam(model_classifier.parameters(),lr=1e-5)
# optimizer_all = optim.SGD(model_all.parameters(),lr=0.1)
optimizer.zero_grad()
# optimizer_classifier.zero_grad()
# optimizer_all.zero_grad()

# optimizer = Adam(lr=1e-5)
# optimizer_classifier = Adam(lr=1e-5)
# model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
# model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
# model_all.compile(optimizer='sgd', loss='mae')

#epoch_length = 1000
epoch_length = 1000
num_epochs = int(options.num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 6))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

vis = True

for epoch_num in range(num_epochs):

    print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

    while True:
        try:

            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
                if mean_overlapping_bboxes == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

            X, Y, img_data = next(data_gen_train)
            X = torch.tensor(X).cuda()
            Y_class = np.array(Y[0])
            Y_class = torch.from_numpy(Y_class).cuda()
            Y_reg = np.array(Y[1])
            Y_reg = torch.from_numpy(Y_reg).cuda()

            model.train()
            #loss_rpn = model_rpn(X, Y)
            output_rpn = model(X,"rpn")
            # print(type(output[0]),type(Y))
            # print(output[0].shape,Y.shape)
            loss_rpn = mseloss(output_rpn[0],Y_class.float())
            loss_reg = mseloss(output_rpn[1],Y_reg.float())
            loss_RPN = loss_rpn + loss_reg
            # print(loss_reg)
            # print(loss_rpn)
            # print(loss_RPN)

            output_rpn[0] = output_rpn[0].cpu().detach().numpy()
            output_rpn[1] = output_rpn[1].cpu().detach().numpy()
            
            loss_RPN.backward(retain_graph=True)
            # loss_rpn.backward(retain_graph=True)
            # loss_reg.backward()
            #print("after train")

            #P_rpn = model_rpn.predict_on_batch(X)
            #!!!!!!!!!!!! output[2]?output[0]
            R = roi_helpers.rpn_to_roi(output_rpn[0], output_rpn[1], C, use_regr=True, overlap_thresh=0.7, max_boxes=300)
            #R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, use_regr=True, overlap_thresh=0.7, max_boxes=300)
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)
            
            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))
            print(Y1[0, :, -1])


            if C.num_rois > 1:
                if len(pos_samples) > 0 and len(neg_samples) > 0 :
                    selected_neg_samples = neg_samples.tolist()
                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois - len(selected_neg_samples), replace=True).tolist()
                elif len(pos_samples) == 0 and len(neg_samples) > 0 :
                    selected_pos_samples = []
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
                elif len(pos_samples) > 0 and len(neg_samples) == 0 :
                    selected_neg_samples = []
                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois - len(selected_neg_samples), replace=True).tolist()

            # if C.num_rois > 1:
            #     if len(pos_samples) < C.num_rois//2:
            #         selected_pos_samples = pos_samples.tolist()
            #     else:
            #         if len(pos_samples) > 0:
            #             selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
            #         else:
            #             selected_pos_samples = []
            #     try:
            #         if len(neg_samples) > 0:
            #             selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
            #         else:
            #             selected_neg_samples = []
            #     except:
            #         if len(neg_samples) > 0:
            #             selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
            #         else:
            #             selected_neg_samples = []
                    
                print("len(neg_samples) is:",len(neg_samples))
                print("len(pos_samples) is:",len(pos_samples))
                print("selected_pos_samples is:",torch.tensor(selected_pos_samples).size())
                print("selected_neg_samples is:",torch.tensor(selected_neg_samples).size())
                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)

            #print("sel_samples is :",sel_samples)
            #print("output_rpn[2] is :",output_rpn[2].size())
            #print("X2 is :",torch.tensor(X2).size())
            #print("X2[:, sel_samples, :] is :",torch.tensor(X2[:, sel_samples, :]).size())
            print("sel_samples is:",torch.tensor(sel_samples).size())
            output_classifer = model([output_rpn[2], X2[:, sel_samples, :]],"classifier")
            #print("finish np.where(Y1[0, :, -1] == 0)")

            Y1 = torch.tensor(Y1).cuda()
            Y2 = torch.tensor(Y2).cuda()
            #print(output_classifer[1].size())
            #print(Y2.size())
            #print(Y2[:, sel_samples, :])
            #print(Y2[:, sel_samples, :].unsqueeze(0).size())
            loss_rpn_classifer = mseloss(output_classifer[0],Y1[:, sel_samples, :].squeeze(0).float())
            loss_reg_classifer = mseloss(output_classifer[1],Y2[:, sel_samples, :].squeeze(0).float())
            loss_classifer = loss_rpn_classifer + loss_reg_classifer * 10
            #print("after train")
            #output_classifer[0] = output_classifer[0].cpu().detach().numpy()
            #output_classifer[1] = output_classifer[1].cpu().detach().numpy()
            total_correct = 0
            total_correct_sign = 0
            epoch_acc_sign = 0.0 

            #print(Y1[:, sel_samples, :].squeeze(0))
            _,predictions = torch.max(output_classifer[0],dim=1)
            _,gt = torch.max(Y1[:, sel_samples, :].squeeze(0),dim=1)
            total_correct += torch.sum(predictions==gt)
            epoch_acc = total_correct.double() / len(gt)
            #print(Y1[:, sel_samples, :].squeeze(0))

            #交通标志识别正确率            
            index_sign_gt = list(torch.nonzero(gt==0))
            # print(index_sign_gt)
            index_sign_pred = list(torch.nonzero(predictions==0))
            # print("index_sign_gt is :",index_sign_gt)
            # print("index_sign_pred is :",index_sign_pred)
            # index_sign = torch.tensor(list(set(index_sign_gt).union(set(index_sign_pred))))
            index_sign = list(set(index_sign_gt).union(set(index_sign_pred)))
            # print("calculate traffic sign acc")
            # print(index_sign)
            if len(index_sign) == 0:
                epoch_acc_sign = 0
            else:
                # print("detect traffic sign")
                # print("index_sign_gt is :",index_sign_gt)
                # print(predictions[index_sign_gt])
                if len(index_sign_gt) == 0:
                    epoch_acc_sign = 0
                else:
                    # if predictions[index_sign_gt]==0 :
                    #     print("predictions[index_sign_gt]==0 is true")
                    # elif predictions[index_sign_gt]==1:
                    #     print("predictions[index_sign_gt]==1 is true")
                    total_correct_sign += torch.sum(predictions[torch.tensor(index_sign_gt).long()]==0)
                    print(total_correct_sign.double() , len(index_sign))
                    epoch_acc_sign = total_correct_sign.double() / len(index_sign)

            loss_classifer.backward()
            optimizer.step()
            #loss_class = model_classifier([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            losses[iter_num, 0] = loss_rpn
            losses[iter_num, 1] = loss_reg

            losses[iter_num, 2] = loss_rpn_classifer
            losses[iter_num, 3] = loss_reg_classifer
            losses[iter_num, 4] = epoch_acc
            losses[iter_num, 5] = epoch_acc_sign

            iter_num += 1
            print('Iter {}/{}'.format(iter_num, epoch_length))

            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])
                traffic_sign_acc = np.mean(losses[:, 5]) 
                traffic_sign_num = len(index_sign)

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if C.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Classifier accuracy for traffic sign bounding boxes from RPN: {} for {} traffic sign bounding boxes'.format(traffic_sign_acc,traffic_sign_num))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    if C.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                    best_loss = curr_loss
                    #model.train(False)
                    #loss_rpn = model_rpn(X, Y)
                    #output = model(X,"all")
                    torch.save(model.state_dict(), C.model_path)
                    #model.save_weights(C.model_path)

                break
        
        except Exception as e:
            print('Exception: {}'.format(e))
            continue

print('Training complete, exiting.')
