import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

# , y = load_iris(return_X_y=True)
x = np.load('inst.npy')
y = np.load('train_flag.npy')
x,y = shuffle(x,y)
clf = LogisticRegression(random_state=0).fit(x[:680], y[:680])

cnt = 0
for i in range(0,680):
	out = clf.predict(x[i].reshape(1,6))
	if out == y[i]:
		cnt +=1

print(cnt/680)
print(clf.coef_)