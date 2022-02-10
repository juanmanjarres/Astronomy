from matplotlib import pyplot as plt
import numpy as np


data = np.zeros( (512,512,3), dtype=np.float64)
data[256,256] = [0.0, 1.0, 0.0]

print(data)
plt.imshow(data, interpolation='nearest')
plt.show()
