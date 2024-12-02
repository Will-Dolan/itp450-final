import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.Transformer import Transformer

from data.Data import Data

""" 
TODO: 
- make all errored variables configurable
- load model separate from train
"""

class Pipeline:
	def __init__(self, args):
		self.args = args
		# TODO: configurable datapath 

		# configurable params

	def configure_gpus(self):
		pass

	def load_dataset(self):
		batch_size=32
		self.seq_size=512
		self.dataset = Data(batch_size, self.seq_size)
		self.dataset.load_dataset()
		self.vocab_size = self.dataset.encode_data()
		self.dataset.split_data()

	def load_model(self):
		pass

	def train_model(self):
		torch.cuda.empty_cache()

		# Finetuning parameters to improve training loss
		embed_dim=512
		n_heads=8
		n_layers=8
		dropout=0.1 # hardcoded in MHA as 0
		learning_rate = 0.0005
		n_epoch = 2000
		print_interval=100
		eval_iters=200 # not used

		if torch.cuda.is_available():
			self.device = 'cuda'
		else: 
			self.device ='cpu'
		print('Running on', self.device)

		# Instantiating the Transformer module
		self.model = Transformer(embed_dim, n_heads, self.vocab_size, self.seq_size, n_layers, self.device)

		self.model = self.model.to(self.device)

		# Print the number of parameters in the model
		print(f"num trainable params = {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
		print('Parameters size:', len(list(self.model.parameters())))

		# Optimizer
		optimizer = torch.optim.AdamW(self.model.parameters(),lr=learning_rate)
		losses = {'train': 0, 'val': 0}

		# Training Loop
		start=time.time()
		for epoch in range(n_epoch):
			self.model.train()
			optimizer.zero_grad()
			
			# get a batch of data
			context,target=self.dataset.get_batch('train')
			context.to(self.device)
			target.to(self.device)

			# pass through the model (context,target)
			_, train_losses = self.model(context, target)
			losses['train'] = train_losses
			train_losses.backward()
			optimizer.step()

			with torch.no_grad():
				val_context, val_target = self.dataset.get_batch('val')
				self.model.eval()
				_, val_losses = self.model(val_context, val_target)
				losses['val'] = val_losses


			if epoch % print_interval == 0 or epoch == n_epoch - 1 or epoch == 0:
				## HOMEWORK: Calculate the training and validation loss using the estimate_loss function
				print(f"[{(time.time()-start):.2f}s] step {epoch}: train loss {losses['train']}, val loss {losses['val']}")
    
		print(f'Training took {time.time()-start} seconds')

	def test_model(self):
		start=time.time()
		context  = torch.zeros((1, 1), dtype=torch.long, device=self.device)
		response = self.model.generation(context, max_tokens=1000)[0].tolist()
		print(f'Inference took {time.time()-start} seconds')
		print('---')
		print(self.dataset.decode(response))
