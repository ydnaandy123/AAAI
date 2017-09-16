import os
from glob import glob
import numpy as np
import scipy.io as sio
import scipy.ndimage


class AAAIParser:
    def __init__(self, dataset_dir, target_height=256, target_width=256):
        self.target_height, self.target_width = target_height, target_width
        self.dataset_dir = dataset_dir
        self.test_name_list = ['BLURBG_Dnn', 'BSDDCU_Dnn', 'ECSSD_Dnn',
                               'GrabCut_Dnn', 'MSRA10K_Dnn', 'PHONE_Dnn']
        self.test_name = self.test_name_list[4]
        self.mat_train_dir = self.dataset_dir + '/new_train'
        self.mat_valid_dir = self.dataset_dir + '/new_validation'
        self.mat_test_dir = self.dataset_dir + '/{}'.format(self.test_name)

        self.mat_train_paths, self.mat_valid_paths, self.mat_test_paths = [], [], []

    def load_mat_train_paths(self):
        self.mat_train_paths = sorted(glob(os.path.join(self.mat_train_dir, "*.mat")))
        return self

    def load_mat_train_ss_paths(self):
        b1s1 = glob(os.path.join(self.mat_train_dir, "*(B1-S1).mat"))
        b2s2 = glob(os.path.join(self.mat_train_dir, "*(B2-S2).mat"))
        b3s3 = glob(os.path.join(self.mat_train_dir, "*(B3-S3).mat"))
        self.mat_train_paths.extend(b1s1)
        self.mat_train_paths.extend(b2s2)
        self.mat_train_paths.extend(b3s3)
        self.mat_train_paths = sorted(self.mat_train_paths)
        return self

    def load_mat_train_w_paths(self):
        b2s2 = glob(os.path.join(self.mat_train_dir, "*(B2-S2).mat"))
        b3s3 = glob(os.path.join(self.mat_train_dir, "*(B3-S3).mat"))
        self.mat_train_paths.extend(b2s2)
        self.mat_train_paths.extend(b3s3)
        self.mat_train_paths = sorted(self.mat_train_paths)
        return self

    def load_mat_train_wo_paths(self):
        b2s2 = glob(os.path.join(self.mat_train_dir, "*(B1-S2).mat"))
        b3s3 = glob(os.path.join(self.mat_train_dir, "*(B1-S3).mat"))
        self.mat_train_paths.extend(b2s2)
        self.mat_train_paths.extend(b3s3)
        self.mat_train_paths = sorted(self.mat_train_paths)
        return self

    def load_mat_train_dd_paths(self):
        self.mat_train_paths = sorted(glob(os.path.join(
            self.mat_train_dir, "*(B1-S*).mat")))
        return self

    def load_mat_valid_paths(self):
        self.mat_valid_paths = sorted(glob(os.path.join(self.mat_valid_dir, "*.mat")))
        return self

    def load_mat_test_paths(self):
        self.mat_test_paths = sorted(glob(os.path.join(self.mat_test_dir, "*.mat")))
        return self

    def load_mat_train_datum_batch(self, start, end):
        print('loading training datum batch...')
        batch_len = end - start
        mat_train_paths_batch = self.mat_train_paths[start:end]
        x_batch, y_batch = [], []
        for idx, mat_train_path in enumerate(mat_train_paths_batch):
            mat_contents = sio.loadmat(mat_train_path)
            x, y = mat_contents['sample'][0][0]['RGBSD'], mat_contents['sample'][0][0]['GT']
            if idx >= batch_len // 2:
                x = np.fliplr(x)
                y = np.fliplr(y)
            x_batch.append(x)
            y_batch.append(y)
        return x_batch, y_batch

    def load_mat_train_datum_batch_aug(self, start, end):
        print('loading training datum batch...')
        mat_train_paths_batch = self.mat_train_paths[start:end]
        x_batch, y_batch = [], []
        for idx, mat_train_path in enumerate(mat_train_paths_batch):
            mat_contents = sio.loadmat(mat_train_path)
            x, y = mat_contents['sample'][0][0]['RGBSD'], mat_contents['sample'][0][0]['GT']
            x, y = self.data_augmentation(x=x, y=y)
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
            x_batch.append(x)
            y_batch.append(y)
        return x_batch, y_batch

    def data_augmentation(self, x, y):
        coin = np.random.randint(0, 2)
        """
        Flip
        """
        if coin == 1:
            x = np.fliplr(x)
            y = np.fliplr(y)

        """
        Color
        """
        x = x.astype(np.float32)
        x[:, :, 0] *= np.random.uniform(0.6, 1.4)
        x[:, :, 1] *= np.random.uniform(0.6, 1.4)
        x[:, :, 2] *= np.random.uniform(0.6, 1.4)
        x[:, :, :3] = np.minimum(x[:, :, :3], 255)
        x = x.astype(np.uint8)

        """
        Rotation
        """
        angle = np.random.uniform(-30.0, 30.0)
        x = scipy.ndimage.interpolation.rotate(input=x, angle=angle, axes=(1, 0))
        y = scipy.ndimage.interpolation.rotate(input=y, angle=angle, axes=(1, 0))

        '''
        """
        Zoom
        """
        zoom = np.random.uniform(1.2, 1.2)
        x = scipy.ndimage.interpolation.zoom(input=x, zoom=zoom)
        y = scipy.ndimage.interpolation.zoom(input=y, zoom=zoom)
        '''

        """
        Crop
        """
        height, width, channel = np.shape(x)
        off_h, off_w = 0, 0
        padding = height - self.target_height
        if padding != 0:
            off_h = np.random.randint(0, padding)
            off_w = np.random.randint(0, padding)
        x = x[off_h:off_h+self.target_height, off_w:off_w+self.target_width, :]
        y = y[off_h:off_h+self.target_height, off_w:off_w+self.target_width]

        return x, y
