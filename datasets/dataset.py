import os
import io
import pickle
import string
import random

import numpy as np
import lmdb
#import cv2
from PIL import Image


class LMDB:
    def __init__(self, path, transform, return_path=False):
        self.env = lmdb.open(path, map_size=1099511627776, readonly=True, lock=False)
        #with self.env.begin(write=False) as txn:
        #    self.image_names= [key.decode() for key in txn.cursor().iternext(keys=True, values=False)]

        cache_file = '_cache_' + ''.join(c for c in path if c in string.ascii_letters)
        cache_path = os.path.join(path, cache_file)
        if os.path.isfile(cache_path):
            self.image_names = pickle.load(open(cache_path, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.image_names = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.image_names, open(cache_path, "wb"))
        self.transform = transform
        self.return_path = return_path
        #self.idxes = list(range(len(self)))

    def return_path(self, isreturn):
        self.return_path = isreturn

    def __len__(self):
        return len(self.image_names)

    def class_id(self, path):
        NotImplementedError


    #def __getitem__(self, idx):
    #    NotImplementedError
    def __getitem__(self, idx):
        #idx = random.choice(self.range)
        #idx = self.idxes[idx]
        image_name = self.image_names[idx]
        with self.env.begin(write=False) as txn:
            image_data = txn.get(image_name)
            #image = np.frombuffer(image_data, np.uint8)
            #image = cv2.imdecode(image, -1)
            image = Image.open(io.BytesIO(image_data))
            label = self.class_id(image_name.decode())


        #image.save('ori.png')
        #import torch
        #import time
        #torch.manual_seed(time.time())

        if self.transform:
            image = self.transform(image)

        #image.save('test.png')
        #import sys; sys.exit()

        if self.return_path:
            return image, label, image_name.decode()
        else:
            return image, label
        #image_name = self.image_names[idx]

        #with self._env.begin(write=False) as txn:
        #    image_data = txn.get(image_name.encode())
        #    print(image_data)
        #    return

class CRC(LMDB):
    def __init__(self, path, transform=None, return_path=False):
        super().__init__(path=path, transform=transform, return_path=return_path)

    def class_id(self, path):
        #print(path)
        grade_id = path.split('_grade_')[1][0]
        #print(grade_id)
        return int(grade_id) - 1

class ExtendedCRC(LMDB):
    def __init__(self, path, transform=None, return_path=False):
        super().__init__(path=path, transform=transform, return_path=return_path)

    def class_id(self, path):
        grade_id = path.split('_grade_')[1][0]
        #print(grade_id)
        return int(grade_id) - 1



#dataset = ExtendedCRC('/home/baiyu/ViT-pytorch/data/ExtendedCRC_LMDB/fold_1')

#for i, name in dataset:
#    i, name
