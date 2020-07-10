import torch
import sys
from gridmask import GridMask

class Trainer():
	def __init__(self, model, train_data, val_data, criterion, optimizer, max_epochs, device, path, cut_output = False,
	 recurrent_model=False, patience=5, grid_mask=None, is_reconstruction = False, lilw=False):
		self.model = model
		self.train_data = train_data
		self.val_data = val_data
		self.criterion = criterion
		self.max_epochs = max_epochs
		self.device = device
		self.optimizer = optimizer
		self.cut_output = cut_output
		self.path = path
		self.grid = None
		self.lilw = lilw
		self.is_reconstruction = is_reconstruction
		self.recurrent_model = recurrent_model
		self.earlyStop = EarlyStop(patience, self.path)
		if (grid_mask is not None):
			self.grid = GridMask(grid_mask['d1'], grid_mask['d2'], device, grid_mask['ratio'], grid_mask['max_prob'], grid_mask['max_epochs'])

	def train_evaluate(self):
		train_losses = []
		val_losses = []
		if (self.grid is not None):
			#self.grid.set_prob(epoch)
			print(self.grid.get_prob())
		for epoch in range(self.max_epochs):
			self.train(train_losses)
			print('Train - Epoch %d, Epoch Loss: %f' % (epoch, train_losses[epoch]))
			self.evaluate(val_losses)
			print('Val Avg. Loss: %f' % (val_losses[epoch]))
			if (torch.cuda.is_available()):
				torch.cuda.empty_cache()
			if (self.earlyStop.check_stop_condition(epoch, self.model, self.optimizer, val_losses[epoch])):
				break
		return train_losses, val_losses

	def train(self, train_losses):
		train_loss = self.model.train()
		epoch_train_loss = 0.0
		for i, (x, y, removed) in enumerate(self.train_data):
			x,y, removed = x.to(self.device), y.to(self.device), removed.to(self.device)
			if (self.grid is not None):
				x_grid = self.grid(x)
			self.optimizer.zero_grad()
			x_in = x if self.grid == None else x_grid
			if (self.recurrent_model):
				if (self.is_reconstruction):
						states_fwd = self.init_hidden(x.size()[0], x.size()[3]*x.size()[4])
						states_bckwd = self.init_hidden(x.size()[0], x.size()[3]*x.size()[4])
						if (self.lilw):
							output = self.model(x, states_fwd, states_bckwd, removed, original_x = x)
						else:
							output = self.model(x, states_fwd, states_bckwd, removed)
				else:
					states = self.init_hidden(x.size()[0], x.size()[3]*x.size()[4])
					if (self.lilw):
						output = self.model(x, states, original_x = x)
					else:
						output = self.model(x,states)
			else:
				output = self.model(x_in)
			#batch : channel : time-steps : lat : lon
			if (self.cut_output and not self.recurrent_model):
				loss = self.criterion(output[:,:,0,:,:], y)
			else:
				loss = self.criterion(output, y, removed)
			loss.backward()
			self.optimizer.step()
			epoch_train_loss += loss.detach().item()
		avg_epoch_loss = epoch_train_loss/len(self.train_data)
		train_losses.append(avg_epoch_loss)

	def evaluate(self, val_losses):
		epoch_val_loss = 0.0
		self.model.eval()
		with torch.no_grad():
			for i, (x, y, removed) in enumerate(self.val_data):
				x,y, removed = x.to(self.device), y.to(self.device), removed.to(self.device)
				if (self.recurrent_model):
					if (self.is_reconstruction):
						states_fwd = self.init_hidden(x.size()[0], x.size()[3]*x.size()[4])
						states_bckwd = self.init_hidden(x.size()[0], x.size()[3]*x.size()[4])
						output = self.model(x, states_fwd, states_bckwd, removed)
					else:
						states = self.init_hidden(x.size()[0], x.size()[3]*x.size()[4])
						output = self.model(x,states)
				else:
					output = self.model(x)
				if (self.cut_output and not self.recurrent_model):
					loss = self.criterion(output[:,:,0,:,:], y)
				else:
					loss = self.criterion(output, y, removed)
				epoch_val_loss += loss.detach().item()
		avg_val_loss = epoch_val_loss/len(self.val_data)
		val_losses.append(avg_val_loss)

	def load_model(self):
		checkpoint = torch.load(self.path)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		loss = checkpoint['loss']
		return self.model, self.optimizer, epoch, loss

	def init_hidden(self, batch_size, hidden_size):
		h = torch.zeros(batch_size,hidden_size, device=self.device)
		return (h,h)



class EarlyStop:
	def __init__(self, threshold, path):
		self.min_loss = sys.float_info.max
		self.count = 0
		self.threshold = threshold
		self.path = path
        
	def check_stop_condition(self, epoch, model, optimizer, loss):
		if (loss < self.min_loss):
			self.save_model(epoch, model, optimizer, loss)
			self.min_loss = loss
			self.count = 0
			return False
		else:
			self.count += 1
			if (self.count >= self.threshold):
				return True
			return False

	def save_model(self, epoch, model, optimizer, loss):
		torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': loss,
			}, self.path)
		print ('=> Saving a new best')

class Tester():
	def __init__(self, model, test_data, criterion, optimizer, device, cut_output, recurrent_model=False):
		self.model = model
		self.test_data = test_data
		self.criterion = criterion
		self.device = device
		self.optimizer = optimizer
		self.cut_output = cut_output
		self.recurrent_model = recurrent_model

	def test(self):
		batch_test_loss = 0.0
		self.model.eval()
		with torch.no_grad():
			for i, (x, y) in enumerate(self.test_data):
				x,y = x.to(self.device), y.to(self.device)
				if (self.recurrent_model):
					states = self.init_hidden(x.size()[0], x.size()[3]*x.size()[4])
					output = self.model(x, states)
				else:
					output = self.model(x)
				if (self.cut_output and not self.recurrent_model):
					loss = self.criterion(output[:,:,0,:,:], y[:,:,0,:,:])
				else:
					loss = self.criterion(output, y)
				batch_test_loss += loss.detach().item()
		test_loss = batch_test_loss/len(self.test_data)
		return test_loss

	def init_hidden(self, batch_size, hidden_size):
		h = torch.zeros(batch_size,hidden_size, device=self.device)
		return (h,h)