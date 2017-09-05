import os
import scipy.misc
from glob import glob
import numpy as np
import scipy.io as sio
import tensorflow as tf


class CityscapesParser:
    def __init__(self, dataset_dir):
        self.cityLabel = [
                            (  0,  0,  0),
                            (  0,  0,  0),
                            (  0,  0,  0),
                            (  0,  0,  0),
                            (  0,  0,  0),
                            (111, 74,  0),
                            ( 81,  0, 81),
                            (128, 64,128),
                            (244, 35,232),
                            (250,170,160),
                            (230,150,140),
                            ( 70, 70, 70),
                            (102,102,156),
                            (190,153,153),
                            (180,165,180),
                            (150,100,100),
                            (150,120, 90),
                            (153,153,153),
                            (153,153,153),
                            (250,170, 30),
                            (220,220,  0),
                            (107,142, 35),
                            (152,251,152),
                            ( 70,130,180),
                            (220, 20, 60),
                            (255,  0,  0),
                            (  0,  0,142),
                            (  0,  0, 70),
                            (  0, 60,100),
                            (  0,  0, 90),
                            (  0,  0,110),
                            (  0, 80,100),
                            (  0,  0,230),
                            (119, 11, 32),
                            (  0,  0,142)
                        ]
        self.img_dir = dataset_dir + '/leftImg8bit_trainvaltest/leftImg8bit'
        self.img_train_dir = self.img_dir + '/train'
        self.img_test_dir = self.img_dir + '/test'
        self.img_valid_dir = self.img_dir + '/val'

        self.img_train_path, self.img_test_path, self.img_valid_path = [], [], []

        self.valid_img, self.valid_label = [], []

    def load_train_image_path(self):
        for folder in os.listdir(self.img_train_dir):
            path = os.path.join(self.img_train_dir, folder, "*_leftImg8bit.png")
            self.img_train_path.extend(glob(path))
        return self

    def load_test_image_path(self):
        for folder in os.listdir(self.img_test_dir):
            path = os.path.join(self.img_test_dir, folder, "*_leftImg8bit.png")
            self.img_test_path.extend(glob(path))
        return self

    def load_valid_image_path(self):
        for folder in os.listdir(self.img_valid_dir):
            path = os.path.join(self.img_valid_dir, folder, "*_leftImg8bit.png")
            self.img_valid_path.extend(glob(path))
        return self

    def load_valid_image(self):
        print('load_valid_image')
        total_len = len(self.img_valid_path)
        for img_valid_path_idx, img_valid_path_cur in enumerate(self.img_valid_path):
            print('[{:d}/{:d}]'.format(img_valid_path_idx, total_len))
            img = scipy.misc.imread(img_valid_path_cur)
            label_valid_path_cur = img_valid_path_cur.replace("leftImg8bit_trainvaltest", "gtFine_trainvaltest")
            label_valid_path_cur = label_valid_path_cur.replace("_leftImg8bit", "_gtFine_labelIds")
            label_valid_path_cur = label_valid_path_cur.replace("leftImg8bit", "gtFine")
            label = scipy.misc.imread(label_valid_path_cur)

            self.valid_img.append(img)
            self.valid_label.append(label)
            break
        return self

    def label_visualize(self, label):
        label = np.array(label.astype(np.uint8), dtype=np.uint8)
        visual_r = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
        visual_g = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
        visual_b = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)

        for i in range(0, len(self.cityLabel)):
            index = np.nonzero(label == i)
            visual_r[index] = self.cityLabel[i][0]
            visual_g[index] = self.cityLabel[i][1]
            visual_b[index] = self.cityLabel[i][2]

        return np.dstack((visual_r, visual_g, visual_b))


class AAAIParser:
    def __init__(self, dataset_dir, target_height=256, target_width=256):
        self.target_height, self.target_width = target_height, target_width
        self.dataset_dir = dataset_dir
        self.mat_train_dir = self.dataset_dir + '/validation'  # train
        self.mat_valid_dir = self.dataset_dir + '/validation'
        self.mat_test_dir = self.dataset_dir + '/test'

        self.mat_train_paths, self.mat_valid_paths, self.mat_test_paths = [], [], []

        self.valid_x, self.valid_y = [], []

    def load_mat_train_paths(self):
        self.mat_train_paths = sorted(glob(os.path.join(self.mat_train_dir, "*.mat")))
        return self

    def load_mat_valid_paths(self):
        self.mat_valid_paths = sorted(glob(os.path.join(self.mat_valid_dir, "*.mat")))
        return self

    def load_mat_train_datum_batch(self, start, end):
        print('loading training datum batch...')
        mat_train_paths_batch = self.mat_train_paths[start:end]
        x_batch, y_batch = [], []
        for idx, mat_train_path in enumerate(mat_train_paths_batch):
            mat_contents = sio.loadmat(mat_train_path)
            x, y = mat_contents['sample'][0][0]['RGBSD'], mat_contents['sample'][0][0]['GT']
            x_rgb = scipy.misc.imresize(x[:, :, :3], (self.target_height, self.target_width), interp='bilinear')
            x_s = scipy.misc.imresize(x[:, :, 3], (self.target_height, self.target_width), interp='bilinear')
            x_d = scipy.misc.imresize(x[:, :, 4], (self.target_height, self.target_width), interp='bilinear')
            x = np.dstack((x_rgb, x_s, x_d))
            y = scipy.misc.imresize(y, (self.target_height, self.target_width), interp='nearest')
            x_batch.append(x)
            y_batch.append(y)
        return x_batch, y_batch

    def load_mat_valid_datum_batch(self, start, end):
        print('loading valid datum batch...')
        mat_valid_paths_batch = self.mat_valid_paths[start:end]
        x_batch, y_batch = [], []
        for idx, mat_valid_path in enumerate(mat_valid_paths_batch):
            mat_contents = sio.loadmat(mat_valid_path)
            x, y = mat_contents['sample'][0][0]['RGBSD'], mat_contents['sample'][0][0]['GT']
            x_rgb = scipy.misc.imresize(x[:, :, :3], (self.target_height, self.target_width), interp='bilinear')
            x_s = scipy.misc.imresize(x[:, :, 3], (self.target_height, self.target_width), interp='bilinear')
            x_d = scipy.misc.imresize(x[:, :, 4], (self.target_height, self.target_width), interp='bilinear')
            x = np.dstack((x_rgb, x_s, x_d))
            y = scipy.misc.imresize(y, (self.target_height, self.target_width), interp='nearest')
            x_batch.append(x)
            y_batch.append(y)
        return x_batch, y_batch

    def visual(self):
        '''
        y_visual = y * 255
        y_visual = np.dstack((y_visual, y_visual, y_visual))
        x_s = x[:, :, 3]
        x_s_visual = np.dstack((x_s, x_s, x_s))
        x_d = x[:, :, 4]
        x_d_visual = np.dstack((x_d, x_d, x_d))
        visual_1 = np.hstack((x[:, :, :3], x_s_visual))
        visual_2 = np.hstack((x_d_visual, y_visual))
        scipy.misc.imshow(np.vstack((visual_1, visual_2)))
        '''
        return self

