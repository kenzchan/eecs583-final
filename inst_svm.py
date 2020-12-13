import numpy as np
from sklearn import svm
from sklearn.utils import shuffle
# , y = load_iris(return_X_y=True)
x = np.load('inst.npy')
y = np.load('train_flag.npy')
x,y = shuffle(x,y)
# y = shuffle(y)
clf = svm.SVC().fit(x[:680], y[:680])

cnt = 0
for i in range(0, 680):
	out = clf.predict(x[i].reshape(1,6))
	if y[i] == out:
		cnt +=1

print(cnt/680)