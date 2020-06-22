import torch
from torch.utils.data import Dataset
import numpy as np

class NCDFDatasets():
	def __init__(self, data, val_split, test_split, cut_y=False,  data_type='Prediction'):
		self.train_data = NCDFDataset(data, test_split, val_split, data_type, False, False, cut_y)
		self.val_data = NCDFDataset(data, test_split, val_split, data_type, False, True, cut_y)
		self.test_data = NCDFDataset(data, test_split, val_split, data_type, True, False, cut_y)

	def get_train(self):
		return self.train_data
	def get_val(self):
		return self.val_data
	def get_test(self):
		return self.test_data

class NCDFDataset(Dataset):
	def __init__(self, data, test_split, val_split, data_type, is_test=False, is_val=False, cut_y=False):
		super(NCDFDataset, self).__init__()
		self.cut_y = cut_y
		self.reconstruction = True if data_type == 'Reconstruction' else False 

		splitter = DataSplitter(data, test_split, val_split)
		if (is_test):
			dataset = splitter.split_test()
		elif (is_val):
			dataset = splitter.split_val()
		else:
			dataset = splitter.split_train()

		#batch, channel, time, lat, lon
		self.x = torch.from_numpy(dataset.x.values).float().permute(0, 4, 1, 2, 3)
		if (self.cut_y):
			self.y = torch.from_numpy(dataset.y.values).float().permute(0, 4, 1, 2, 3)[:,:,0,:,:]
		else:
			self.y = torch.from_numpy(dataset.y.values).float().permute(0, 4, 1, 2, 3)
		del dataset

		if (self.reconstruction):
			data_cat = torch.cat((self.x, self.y), 2)
			self.y = data_cat.clone().detach()
			self.x, self.removed = self.removeObservations(data_cat.clone().detach())

	def __getitem__(self, index):
		if (self.reconstruction):
			return (self.x[index,:,:,:,:], self.y[index,:,:,:,:], self.removed[index])
		elif (self.cut_y):
			return (self.x[index,:,:5,:,:], self.y[index,:,:,:])
		else:
			return (self.x[index,:,:5,:,:], self.y[index,:,:,:,:])

	def __len__(self):
		return self.x.shape[0]

	def removeObservations(self, data):
		removed_observations = torch.zeros(data.shape[0], dtype=torch.long)
		for i in range(data.shape[0]):
			index = np.random.randint(0, data.shape[2])
			data[i,:,index,:,:] *= 0
			removed_observations[i] = index
		return data, removed_observations



class DataSplitter():
	def __init__(self, data, val_split=0, test_split=0):
		self.val_split = val_split
		self.test_split = test_split
		self.data = data

	def split_train(self):
		test_cutoff = int(self.data.sample.size * self.test_split)
		val_cutoff = int(self.data.sample.size * self.val_split)
		return self.data[dict(sample=slice(0, self.data.sample.size - val_cutoff - test_cutoff))]

	def split_val(self):
		test_cutoff = int(self.data.sample.size * self.test_split)
		val_cutoff = int(self.data.sample.size * self.val_split)
		return self.data[dict(sample=slice(self.data.sample.size - val_cutoff - test_cutoff, self.data.sample.size - test_cutoff))] 

	def split_test(self):
		test_cutoff = int(self.data.sample.size * self.test_split)
		return self.data[dict(sample=slice(self.data.sample.size - test_cutoff, None))]


