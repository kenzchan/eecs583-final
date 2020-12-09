import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
# , y = load_iris(return_X_y=True)
x = np.load('inst.npy')
y = np.load('CPU_runtime.npy')
# print(x.shape)
x = x.T
# print(x[0])
# print(y)
x,y = shuffle(x,y)
clf = LinearRegression().fit(x[:600], y[:600])

cnt = 0
for i in range(600,680):
	out = clf.predict(x[i].reshape(1,6))
	print(out, y[i])
	if out < 0: 
		out = 0
	cnt += abs(out-y[i])
print(cnt/80)
print(clf.coef_)