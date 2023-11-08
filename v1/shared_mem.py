from multiprocessing.shared_memory import SharedMemory
import pandas as pd
import os

class SharedMemoryWriter:
    def __init__(self, sm: SharedMemory):
        self.sm = sm
        self.i = 0
    def write(self, b):
        j = self.i + len(b)
        print('Write', len(b), 'bytes.')
        if j <= self.sm.size:
            self.sm.buf[self.i:j] = b
            self.i = j
            return len(b)
        else:
            details = f'Tried to write {len(b)} bytes but only {self.sm.size - self.i} available.'
            raise IOError(f'Insufficient capacity in buffer.\n{details}')
        
class SharedMemoryReader:
    def __init__(self, sm: SharedMemory):
        self.sm = sm
        self.i = 0

    def read(self, size=None):
        if self.i >= self.sm.size:
            return -1
        if size is None:
            ret = self.sm.buf[self.i:]
            self.i = self.sm.size
            return ret
        else:
            j = self.i + size
            ret = self.sm.buf[self.i:j]
            self.i = j
            return ret
        
    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            self.i = offset
        elif whence == os.SEEK_CUR:
            self.i += offset
        elif whence == os.SEEK_END:
            self.i = self.sm.size + offset
        else:
            raise IOError(f'Got unsupported whence argument {whence}.')
        
    def readline(self):
        raise IOError('readline unimplemented')

"""
## usage

# resource owner

df = pd.DataFrame([[1,2], [3,4]], columns=['a', 'b'])
sm = SharedMemory(create=True, size=1024) # pick right size
df.to_pickle(SharedMemoryWriter(sm))
sm.close()   # later
sm.unlink()  # at end

# resource consumer

from multiprocessing.shared_memory import SharedMemory
from multiprocessing.resource_tracker import unregister

sm = SharedMemory(name='<sm.name>')
df = pd.read_pickle(SharedMemoryReader(sm))
sm.close()
unregister(sm._name, 'shared_memory')  
"""