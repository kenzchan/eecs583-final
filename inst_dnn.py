import numpy as np
from sklearn import svm
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(6, 3)
		self.fc2 = nn.Linear(3, 1)
		self.fc3 = nn.Linear(1, 1)
		self.fc4 = nn.Sigmoid()
	def forward(self, x):
		x = x.float()
		x = self.fc1(x).float()
		x = self.fc2(x)
		x = self.fc3(x)
		return x

class customDataset(Dataset):
	def __init__(self):
		x = np.load('inst.npy')
		y = np.load('train_flag.npy')
		self.x = x[:600]
		self.y = y[:600]
	def __getitem__(self, index):
		return self.x[index], self.y[index]    
	def __len__(self):
		return 600

if __name__ == '__main__':
	# , y = load_iris(return_X_y=True)
	
	net = Net()

	val_data = torch.from_numpy(np.load('inst.npy')[600:680]).float()
	val_flag = torch.from_numpy(np.load('train_flag.npy')[600:680]).float()
	criterion = nn.L1Loss()
	optimizer = optim.Adam(net.parameters(), lr=1e-4)
	train_set = customDataset()
	train_loader = DataLoader(
	dataset=train_set, batch_size=32, shuffle=True)
	for epoch in range(200):  # loop over the dataset multiple times

		running_loss = 0.0
		for inputs, labels in train_loader:
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)
			outputs = outputs.float()
			labels = labels.float()
			outputs = outputs.view(-1)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
		val_output = net(val_data)
		cnt = 0
		for i in range(80):
			if torch.round(val_output[i]) == val_flag[i]:
				cnt += 1
		print('[%d] loss: %.3f' %
			(epoch + 1, running_loss / 680), 'val loss: ' + str(cnt/80))

	print('Finished Training')

	# print(cnt/80)