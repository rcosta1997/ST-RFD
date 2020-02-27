import os
import torch

class Evaluator:
    def __init__(self, model, loss, data_loader, device):
        self.model = model
        self.loss = loss
        self.data_loader = data_loader
        self.device = device
        
    def evaluate(self):
        with torch.no_grad():
            eval_loss = 0.0
            self.model.eval()
            for i, (x, y) in enumerate(self.data_loader):
                x,y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                eval_loss += loss.item()
            avg_loss = eval_loss/len(self.data_loader)
        return avg_loss