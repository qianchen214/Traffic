import torch
import torch.nn.functional as F

# if K.backend() == 'tensorflow':
#     import tensorflow as tf


class RoiPoolingConv():  # only support one feature map input this layer at once
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''
    def __init__(self, pool_size, num_rois, **kwargs):

        #assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        flag = True
        if flag:
            self.nb_channels = 512
        else :
            self.nb_channels = input_shape[3]

    def compute_output_shape(self, input_shape):
        flag = True
        if flag:
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):   # the input is a list whose length is 2

        assert(len(x) == 2)

        img = x[0]
        rois = x[1]

        input_shape = img.shape

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]
            
            row_length = w / float(self.pool_size)
            col_length = h / float(self.pool_size)

            num_pool_regions = self.pool_size

            #NOTE: the RoiPooling implementation differs between theano and tensorflow due to the lack of a resize op
            # in theano. The theano implementation is much less efficient and leads to long compile times
            flag = False
            if flag:
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = x + ix * row_length
                        x2 = x1 + row_length
                        y1 = y + jy * col_length
                        y2 = y1 + col_length

                        x1 = torch.tensor(x1).int()
                        x2 = torch.tensor(x2).int()
                        y1 = torch.tensor(y1).int()
                        y2 = torch.tensor(y2).int()

                        # x1 = torch.IntTensor(x1)
                        # x2 = torch.IntTensor(x2)
                        # y1 = torch.IntTensor(y1)
                        # y2 = torch.IntTensor(y2)

                        one = torch.tensor(1)

                        x2 = x1 + torch.maximum(one,x2-x1)
                        y2 = y1 + torch.maximum(one,y2-y1)
                        
                        new_shape = [input_shape[0], input_shape[1],
                                     y2 - y1, x2 - x1]

                        x_crop = img[:, :, y1:y2, x1:x2]
                        xm = torch.reshape(x_crop, new_shape)
                        pooled_val_2 = torch.max(xm,2)
                        pooled_val_3 = torch.max(xm,3)
                        pooled_val_3 = torch.max(pooled_val_3[0])
                        pooled_val_2 = torch.max(pooled_val_2[0])
                        pooled_val = [pooled_val_3,pooled_val_2]
                        pooled_val = torch.tensor(pooled_val)
                        pooled_val = torch.max(pooled_val)
                        outputs.append(pooled_val)

            else:
                # x = torch.tensor(x).int()
                # y = torch.tensor(y).int()
                # w = torch.tensor(w).int()
                # h = torch.tensor(h).int()

                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                rs = F.interpolate(img[:, :, y:y+h, x:x+w], (self.pool_size, self.pool_size))
                outputs.append(rs)

        #print(outputs)

        final_output = torch.cat(outputs,0)
        final_output = torch.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        flag = True
        if flag:
            final_output = final_output.permute(0,1,4,2,3)
        else:
            final_output = final_output.permute(0,1,2,3,4)
        return final_output.cuda()
