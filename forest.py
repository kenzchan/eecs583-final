import numpy as np
from sklearn.ensemble import RandomForestClassifier
# , y = load_iris(return_X_y=True)
x = np.load('train_data.npy')
y = np.load('train_flag.npy')
print(y[:600])
clf = RandomForestClassifier()
clf = clf.fit(x[:600], y[:600])

cnt = 0
for i in range(600,680):
	out = clf.predict(x[i].reshape(1,128))
	if y[i] == out:
		cnt +=1

print(cnt/80)