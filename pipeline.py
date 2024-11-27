import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.Transformer import Transformer

""" 
TODO: 
- make all errored variables configurable
- load model separate from train
"""

class Pipeline:
	def __init__(self, args):
		self.args = args
		# TODO: configurable datapath 

	def configure_gpus(self):
		pass

	def load_dataset(self):

		pass

	def load_model(self):
		pass

	def train_model(self):
		# Finetuning parameters to improve training loss
		batch_size=16
		seq_size=128
		embed_dim=256
		n_layers=64
		n_epoch = 4000

		# Instantiating the Transformer module
		model = Transformer(embed_dim,n_heads)

		if torch.cuda.is_available():
			device = 'cuda'
		else: 
			device ='cpu'

		model = model.to(device)

		# Print the number of parameters in the model
		print(f"num trainable params = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

		# Optimizer
		optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

		# Training Loop
		start=time.time()
		for epoch in range(n_epoch):

			# Training
			model.train()
			
			# Get a batch of training data
			context,target = get_batch('train')

			optimizer.zero_grad()
			
			# Pass through the model (context,target) and calculate train loss
			y,loss = model(context,target)
			
			# Backward propagation
			loss.backward()

			# Validation
			model.eval()
			with torch.no_grad():    
				# Get a batch of validation data
				context,target = get_batch('val')
				
				# Pass through the model (context,target) and calculate val loss
				y,val_loss = model(context,target)

			# Optimizer step
			optimizer.step()

			if epoch % print_interval == 0 or epoch == n_epoch - 1:
				# Print the training and validation loss (NOTE: estimate_loss function was not created because loss is already calculated in forward function)
				print(f"step {epoch}: train loss {loss}, val loss {val_loss}")

		print(f'Training took {time.time()-start} seconds')

	def test_model(self):
		pass
