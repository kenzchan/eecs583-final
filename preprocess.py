import numpy as np


a = np.load('inst.npy')
b = np.load('case_a_input_idx.npy')
out = np.zeros((b.shape[0],6))
for x in range(b.shape[0]):
	out[x] = a[b[x]]

print(out.shape)
np.save('case_a_inst.npy', out)
