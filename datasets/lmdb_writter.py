import os
import glob

import lmdb
#import sys
#sys.path.append(os.getcwd())
#from slice_dataset import slice_image, save_patches

#import cv2
#def lmdb_write(lmdb_path, )
class LMDB:
    def __init__(self, lmdb_path):
        #self.lmdb_path = lmdb_path
        map_size = 10 << 40
        #print(lmdb_path)
        self.env = lmdb.open(lmdb_path, map_size=map_size)

    def add_files(self, pathes):
        with self.env.begin(write=True) as txn:
            for fp in pathes:
                with open(fp, 'rb') as f:
                    image_buff = f.read()

                basename = os.path.basename(fp)
                txn.put(basename.encode(), image_buff)


#if __name__ == '__main__':
#
#    src_path = '/home/baiyu/test_can_be_del3/fold_3'
#    dest_path = '/home/baiyu/ViT-pytorch/data/ExtendedCRC_LMDB/fold_3'
#    lmdb_writter = LMDB(dest_path)
#    search_path = os.path.join(src_path, '**', '*.png')
#    count = 0
#    res = []
#    for path in glob.iglob(search_path, recursive=True):
#        count += 1
#        res.append(path)
#
#        if count == 10000:
#            print('write to disk...')
#            lmdb_writter.add_files(res)
#            res = []
#            count = 0
#
#    lmdb_writter.add_files(res)
