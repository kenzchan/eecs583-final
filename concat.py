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

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.emb = nn.Embedding(129, 128)
		self.rnn = nn.LSTM(128, 64)
		self.rnn2 = nn.LSTM(64, 64)
		self.bn = nn.BatchNorm1d(64)
		# self.f = nn.Flatten(0, -1)
		self.fc1 = nn.Linear(2048, 1)
		self.fc2 = nn.Sigmoid()
	def forward(self, x):
		# x = x.float()
		# x = self.fc1(x).float()
		# x = self.fc2(x).float()
		tmp = x[:,1024:1030]
		x = self.emb(x[:,:1024])
		x, _ = self.rnn(x)
		# x = x[-1]
		x, _ = self.rnn2(x)
		x = x[-1]
		x = self.bn(x)
		x = torch.reshape(x, (32,2048))
		x = torch.concat((x,tmp), 0)
		x = self.fc1(x)
		# print(x.shape)
		x = self.fc2(x)

		return x

class customDataset(Dataset):
	def __init__(self):
		x = np.load('case_a_input.npy')
		y = np.load('case_a_output.npy')
		z = np.load('case_a_inst.npy')
		out = np.zeros((246,1030))
		for i in range(246):
			out[i] = np.append(x[i],z[i])
		self.x = torch.LongTensor(out[:192])
		self.y = y[:192]
	def __getitem__(self, index):
		return self.x[index], self.y[index]    
	def __len__(self):
		return 192

if __name__ == '__main__':
	# , y = load_iris(return_X_y=True)
	
	net = Net().cpu()

	val_data = torch.LongTensor(np.load('case_a_input.npy')[192:224])
	val_flag = torch.from_numpy(np.load('case_a_output.npy')[192:224]).float()
	criterion = nn.MSELoss()
	optimizer = optim.Adam(net.parameters(), lr=1e-4)
	train_set = customDataset()
	train_loader = DataLoader(
	dataset=train_set, batch_size=32, shuffle=True)
	for epoch in range(100):  # loop over the dataset multiple times

		running_loss = 0.0
		for inputs, labels in train_loader:
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)
			outputs = outputs.float()
			labels = labels.float()
			outputs = outputs.view(-1)
			# print(outputs.shape)
			# print(labels.shape)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
		val_output = net(val_data)
		cnt = 0
		for i in range(32):
			if torch.round(val_output[i]) == val_flag[i]:
				cnt += 1
		print('[%d] loss: %.3f' %
			(epoch + 1, running_loss / 192))
		print('val acc: ' + str(cnt/32))

	print('Finished Training')

	# print(cnt/80)