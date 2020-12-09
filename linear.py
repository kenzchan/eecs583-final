import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle

# , y = load_iris(return_X_y=True)
x = np.load('train_data.npy')
y = np.load('CPU_runtime.npy')
x,y = shuffle(x,y)
clf = LinearRegression().fit(x[:600], y[:600])

cnt = 0
for i in range(600,680):
	out = clf.predict(x[i].reshape(1,128))
	if out == y[i]:
		cnt +=1

print(cnt/80)
print(clf.coef_)