import torch
from torch.utils.data import Dataset

class NCDFDatasets():
	def __init__(self, data, val_split, test_split):
		self.train_data = NCDFDataset(data, test_split, val_split, False, False)
		self.val_data = NCDFDataset(data, test_split, val_split, False, True)
		self.test_data = NCDFDataset(data, test_split, val_split, True, False)

	def get_train(self):
		return self.train_data
	def get_val(self):
		return self.val_data
	def get_test(self):
		return self.test_data

class NCDFDataset(Dataset):
	def __init__(self, data, test_split, val_split, is_test=False, is_val=False):
		super(NCDFDataset, self).__init__()

		splitter = DataSplitter(data, test_split, val_split)
		if (is_test):
			dataset = splitter.split_test()
		elif (is_val):
			dataset = splitter.split_val()
		else:
			dataset = splitter.split_train()

		#batch, channel, time, lat, lon
		self.x = torch.from_numpy(dataset.x.values).float().permute(0, 4, 1, 2, 3)
		self.y = torch.from_numpy(dataset.y.values).float().permute(0, 4, 1, 2, 3)#[:,:,0,:,:]
		del dataset

	def __getitem__(self, index):
		return (self.x[index,:,:5,:,:], self.y[index,:,:,:,:]#self.y[index,:,:,:])

	def __len__(self):
		return self.x.shape[0]



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


