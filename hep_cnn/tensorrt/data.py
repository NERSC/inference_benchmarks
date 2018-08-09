import tensorflow as tf
import numpy as np
import h5py as h5
import os
import multiprocessing
import signal
import itertools

#data handler class
class DataSet(object):
    
    def reset(self):
        self._epochs_completed = 0
        self._file_index = 0
        self._data_index = 0
    
    
    def load_next_file(self):
        #only load a new file if there are more than one file in the list:
        if self._num_files > 1 or not self._initialized:
            try:
                with h5.File(self._filelist[self._file_index],'r', driver='core') as f:
                    #determine total array size:
                    numentries=f['data'].shape[0]
                
                    if self._split_file:
                        blocksize = int(np.ceil(numentries/float(self._num_tasks)))
                        start = self._taskid*blocksize
                        end = (self._taskid+1)*blocksize
                    else:
                        start = 0
                        end = numentries
                
                    #load the chunk which is needed
                    self._images = f['data'][start:end].astype(self._dtype)
                    self._labels = f['label'][start:end].astype(np.int32)
                    self._normweights = f['normweight'][start:end].astype(np.float32)
                    self._weights = f['weight'][start:end].astype(np.float32)
                    self._psr = f['psr'][start:end].astype(np.int32)
                    f.close()
            except EnvironmentError:
                raise EnvironmentError("Cannot open file "+self._filelist[self._file_index])
                
            #sanity checks
            assert self._images.shape[0] == self._labels.shape[0], ('images.shape: %s labels.shape: %s' % (self._images.shape, self_.labels.shape))
            assert self._labels.shape[0] == self._normweights.shape[0], ('labels.shape: %s normweights.shape: %s' % (self._labels.shape, self._normweights.shape))
            assert self._labels.shape[0] == self._psr.shape[0], ('labels.shape: %s psr.shape: %s' % (self._labels.shape, self._psr.shape))
            self._initialized = True
        
            #set number of samples
            self._num_examples = self._labels.shape[0]
            
            self._labels = np.expand_dims(self._labels, axis=1)
            self._normweights = np.expand_dims(self._normweights, axis=1)
            self._weights = np.expand_dims(self._weights, axis=1)
            self._psr = np.expand_dims(self._psr, axis=1)
            
            #transpose images if data format is NHWC
            if self._data_format == "NHWC":
                #transform for NCHW to NHWC
                self._images = np.transpose(self._images, (0,2,3,1))
            
        if self._shuffle:
          #create permutation
          perm = np.arange(self._num_examples)
          np.random.shuffle(perm)
          #shuffle
          self._images = self._images[perm]
          self._labels = self._labels[perm]
          self._normweights = self._normweights[perm]
          self._weights = self._weights[perm]
          self._psr = self._psr[perm]
        
    
    def __init__(self, filelist,num_tasks=1,taskid=0,split_filelist=False,split_file=False,shuffle=False,data_format="NCHW",dtype=np.float32):
        """Construct DataSet"""
        #multinode stuff
        self._num_tasks = num_tasks
        self._taskid = taskid
        self._split_filelist = split_filelist
        self._split_file = split_file
        self._data_format = data_format
        self._dtype = dtype
        self._shuffle = shuffle
        
        #split filelist?
        self._num_files = len(filelist)
        start = 0
        end = self._num_files
        if self._split_filelist:
            self._num_files = int(np.floor(len(filelist)/float(self._num_tasks)))
            start = self._taskid * self._num_files
            end = start + self._num_files
        
        assert self._num_files > 0, ('filelist is empty')
        
        self._filelist = filelist[start:end]
        self._initialized = False
        self.reset()
        self.load_next_file()

    @property
    def num_files(self):
        return self._num_files
    
    @property
    def num_samples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next(self):
      
      for i in itertools.count(1): 
        try:
          images, labels, normweights, weights, psr = self.next_batch(1)
        except:
          return
      
        #squeeze dims:
        images = np.squeeze(images, axis=0)
        labels = np.squeeze(labels, axis=0)
        normweights = np.squeeze(normweights, axis=0)
        weights = np.squeeze(weights, axis=0)
        psr = np.squeeze(psr, axis=0)
        
        yield images, labels, normweights, weights, psr
    
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._data_index
        self._data_index += batch_size
        end=int(np.min([self._num_examples,self._data_index]))
        
        #take what is there
        images = self._images[start:end]
        labels = self._labels[start:end]
        normweights = self._normweights[start:end]
        weights = self._weights[start:end]
        psr = self._psr[start:end]
        
        if self._data_index > self._num_examples:
            #remains:
            remaining = self._data_index-self._num_examples
            
            #first, reset data_index and increase file index:
            self._data_index=0
            self._file_index+=1
            
            #check if we are at the end of the file list
            if self._file_index >= self._num_files:
              #epoch is finished
              self._epochs_completed += 1
              #reset stuff
              self._file_index=0
              if self._shuffle:
                np.random.shuffle(self._filelist)
              return
            
            #load the next file
            self.load_next_file()
            #assert batch_size <= self._num_examples
            #call rerucsively
            tmpimages,tmplabels,tmpnormweights,tmpweights,tmppsr = self.next_batch(remaining)
            #join
            images = np.concatenate([images,tmpimages],axis=0)    
            labels = np.concatenate([labels,tmplabels],axis=0)
            normweights = np.concatenate([normweights,tmpnormweights],axis=0)
            weights = np.concatenate([weights,tmpweights],axis=0)
            psr = np.concatenate([psr,tmppsr],axis=0)
        
        return images, labels, normweights, weights, psr