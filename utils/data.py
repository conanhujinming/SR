import scipy.sparse as sp
import numpy as np

class Dataset(object):

    def __init__(self, path):
        pass
        #self.input_croped
        #self.target_croped
        #self.num_train
        self.index_in_epoch=0
    def next_batch(self,batch_size):
        start=self.index_in_epoch
        self.index_in_epoch+=batch_size
        if(self.index_in_epoch>self.num_train)


