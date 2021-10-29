import os

class DataSetConfig:
    def __init__(self, dataset_name):
        if dataset_name == 'CRC':
            #self.lmdb_path = '/home/baiyu/ViT-pytorch/data/CRC_LMDB'
            self.lmdb_path = 'data/CRC_LMDB'
        elif dataset_name == 'ECRC':
            #self.lmdb_path = '/home/baiyu/ViT-pytorch/data/ExtendedCRC_LMDB/'
            self.lmdb_path = 'data/ExtendedCRC_LMDB/'
        else:
            raise ValueError('does not support this dataset {}'.format(dataset_name))
        self.fold1 = os.path.join(self.lmdb_path, 'fold_1')
        self.fold2 = os.path.join(self.lmdb_path, 'fold_2')
        self.fold3 = os.path.join(self.lmdb_path, 'fold_3')
