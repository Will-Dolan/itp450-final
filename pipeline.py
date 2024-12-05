import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.Transformer import Transformer
from tqdm import tqdm
from data.Data import Data

""" 
TODO: 
- make all errored variables configurable
- load model separate from train
"""

class Pipeline:
	def __init__(self, args):
		self.args = args
		self.seq_size = 128
		self.batch_size = 64 # args.batch_size 
		self.epochs = 20 # args.epochs
		self.saved_model_pathway = args.saved_model_pathway
		self.seed = args.seed
		self.experiment_name = args.experiment_name
		self.init_learning_rate = args.init_learning_rate

		# Ensure reproducibility
		if self.seed is not None:
			torch.manual_seed(self.seed)
			np.random.seed(self.seed)

	def configure_gpus(self):
		pass

	def load_dataset(self):
		val_split = 0.2
		self.dataset = Data(self.batch_size, self.seq_size, num_samples=10000)
		self.dataset.load_dataset()
		self.vocab_size = self.dataset.encode_data()
		self.dataset.split_dataset(val_split)

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
		print(f"Num trainable params = {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
		print('Parameters size:', len(list(self.model.parameters())))

		# Optimizer
		optimizer = torch.optim.AdamW(self.model.parameters(),lr=learning_rate)
		losses = {'train': 0, 'val': 0}

		# Training Loop
		start=time.time()
		for epoch in range(self.epochs):
			self.model.train()
			optimizer.zero_grad()
			print("Epoch " + str(epoch))
			for context, target in tqdm(self.dataset.train_dataloader):
				# context, target=self.dataset.get_batch('train')
				context.to(self.device)
				target.to(self.device)

				# pass through the model (context,target)
				_, train_losses = self.model(context, target)
				losses['train'] = train_losses
				train_losses.backward()
				optimizer.step()

			with torch.no_grad():
				for val_context, val_target in self.dataset.val_dataloader:
					# val_context, val_target = self.dataset.get_batch('val')
					val_context = val_context.to(self.device)
					val_target = val_target.to(self.device)
					_, val_losses = self.model(val_context, val_target)
					losses['val'] = val_losses


			if epoch % print_interval == 0 or epoch == self.epochs - 1 or epoch == 0:
				print(f"[{(time.time()-start):.2f}s] step {epoch}: train loss {losses['train']}, val loss {losses['val']}")
    
		print(f'Training took {time.time()-start} seconds')

	def test_model(self):
		start=time.time()
		context  = torch.zeros((1, 1), dtype=torch.long, device=self.device)
		response = self.model.generation(context, max_tokens=1000)[0].tolist()
		print(f'Inference took {time.time()-start} seconds')
		print('---')
		print(self.dataset.decode(response))
