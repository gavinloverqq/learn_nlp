import numpy, cupy, cupyx

# print( cupyx.get_runtime_info() )


gpu = True
if not gpu:
    xp = numpy
else:
    xp = cupy

mydata = xp.empty((3,), dtype='f')    
mydata_like = xp.zeros_like(mydata)