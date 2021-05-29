import os
import glob
import h5py
import numpy as np
from skimage import io, color, exposure, transform
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

NUM_CLASSES = 43
IMG_SIZE = 48
batch_size = 16

def preprocess_img(img):
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.rollaxis(img,-1)

    return img


def get_class(img_path):
    return int(img_path.split('/')[-2])


def img_process():
    try:
        with  h5py.File('X.h5') as hf: 
            X, Y = hf['imgs'][:], hf['labels'][:]
        print("Loaded images from X.h5")
        
    except (IOError,OSError, KeyError):  
        print("Error in reading X.h5. Processing all images...")
        root_dir = '/home/qianchen/class/Traffic/dataset/GTSRB_Train_images/images/'
        imgs = []
        labels = []

        all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
        #打乱图片路径顺序
        np.random.shuffle(all_img_paths)
        for img_path in all_img_paths:
            try:
                img = preprocess_img(io.imread(img_path))
                
                # io.imread 读入的数据是 uint8
                
                label = get_class(img_path)
                imgs.append(img)
                labels.append(label)

                if len(imgs)%1000 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
            except (IOError, OSError):
                print('missed', img_path)
                pass

        X = np.array(imgs, dtype='float32')
        Y = np.array(labels, dtype='uint8')
       # Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]
        # Y = ***[labels] 生成one-hot编码的方式
        with h5py.File('X.h5','w') as hf:
            hf.create_dataset('imgs', data=X)
            hf.create_dataset('labels', data=Y)

    try:
        with  h5py.File('X_test.h5') as hf: 
            X_test, y_test = hf['imgs'][:], hf['labels'][:]
        print("Loaded images from X_test.h5")
    except (IOError,OSError, KeyError):  
        print("Error in reading X.h5. Processing all images...")

    test = pd.read_csv('/home/qianchen/class/Traffic/dataset/GTSRB_Test_images/GT-final_test.csv',sep=';')

        
    X_test = []
    y_test = []
    i = 0
    for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join('/home/qianchen/class/Traffic/dataset/GTSRB_Test_images/images/',file_name)
        X_test.append(preprocess_img(io.imread(img_path)))
        y_test.append(class_id)

    X_test = np.array(X_test, dtype='float32')
    y_test = np.array(y_test, dtype='uint8')

    with h5py.File('X_test.h5','w') as hf:
        hf.create_dataset('imgs', data=X_test)
        hf.create_dataset('labels', data=y_test)



    index=np.zeros(1307, dtype='int')
    for i in range(1307):
        index[i]=i*30+np.random.randint(0,30) 

    X_val = X[index]
    y_val = Y[index]
    # creat the training index1
    index1=np.setdiff1d(np.array(range(39209)), index, assume_unique=True)
    X_train=X[index1]
    y_train=Y[index1]

    normalize = 0
    # Normalize the data: subtract the mean image
    if normalize:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image


    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test


class mydataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __getitem__(self, index):
        img = self.images[index]
        target = self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)


def load_data():
    X_train, y_train, X_val, y_val, X_test, y_test = img_process()
    train_data = mydataset(X_train, y_train)
    val_data = mydataset(X_val, y_val)
    test_data = mydataset(X_test, y_test)
    trainloader = DataLoader(train_data, batch_size, shuffle=True)
    valloader = DataLoader(val_data, batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size, shuffle=True)
    return trainloader, valloader, testloader
        