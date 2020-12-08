import numpy as np
from sklearn.linear_model import LogisticRegression

# , y = load_iris(return_X_y=True)
x = np.load('train_data.npy')
y = np.load('train_flag.npy')
clf = LogisticRegression(random_state=0).fit(x, y)

cnt = 0
for i in range(680):
	out = clf.predict(x[i].reshape(1,128))
	if out == y[i]:
		cnt +=1

print(cnt/680)