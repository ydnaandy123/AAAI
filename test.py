import dataset_parser
import numpy as np
import scipy.io as sio

aaai_parser = dataset_parser.AAAIParser('./dataset/AAAI',
                                        target_height=256, target_width=256)
aaai_parser.load_mat_valid_paths()

mat_contents = sio.loadmat(aaai_parser.mat_valid_paths[0])

mat_contents['pred'] = np.zeros((256, 256), dtype=np.uint8)
name = aaai_parser.mat_valid_paths[0].split('/')[-1]
sio.savemat(name, {'a_dict': mat_contents})
