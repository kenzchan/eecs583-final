import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

# , y = load_iris(return_X_y=True)
x = np.load('train_data.npy')
y = np.load('train_flag.npy')
x,y = shuffle(x,y)
clf = LogisticRegression(random_state=0).fit(x[:600], y[:600])

cnt = 0
for i in range(600,680):
	out = clf.predict(x[i].reshape(1,128))
	if out == y[i]:
		cnt +=1

print(cnt/80)
print(clf.coef_)