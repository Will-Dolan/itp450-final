import time
import numpy as np
import matplotlib.pyplot as plt
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
		self.epochs = 30 # args.epochs
		self.saved_model_pathway = args.saved_model_pathway
		self.seed = args.seed
		self.experiment_name = args.experiment_name
		self.init_learning_rate = args.init_learning_rate
		self.num_samples = 10000

		# Ensure reproducibility
		if self.seed is not None:
			torch.manual_seed(self.seed)
			np.random.seed(self.seed)

	def configure_gpus(self):
		pass

	def load_dataset(self):
		val_split = 0.2
		start=time.time()
		self.dataset = Data(self.batch_size, self.seq_size, num_samples=self.num_samples)
		self.dataset.load_dataset()
		self.vocab_size = self.dataset.encode_data()
		self.dataset.split_dataset(val_split)
		print(f'Data loading took {time.time()-start} seconds')

	def load_model(self):
		pass

	def train_model(self):
		torch.cuda.empty_cache()

		# Finetuning parameters to improve training loss
		embed_dim=768
		n_heads=12
		n_layers=12
		learning_rate=1e-4
		print_interval=1 # print after every epoch
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
		train_losses = []
		val_losses = []

		# Training Loop
		start=time.time()
		for epoch in range(self.epochs):
			self.model.train()
			optimizer.zero_grad()
			print("Epoch " + str(epoch))
			for context, target in tqdm(self.dataset.train_dataloader):
				context.to(self.device)
				target.to(self.device)

				# pass through the model (context,target)
				_, train_loss = self.model(context, target)
				train_losses.append(train_loss)
				train_loss.backward()
				optimizer.step()

			with torch.no_grad():
				for val_context, val_target in self.dataset.val_dataloader:
					val_context = val_context.to(self.device)
					val_target = val_target.to(self.device)
					_, val_loss = self.model(val_context, val_target)
					val_losses.append(val_loss)

			if epoch % print_interval == 0 or epoch == self.epochs - 1 or epoch == 0:
				print(f"[{(time.time()-start):.2f}s] step {epoch}: train loss {train_loss}, val loss {val_loss}")
    
		print(f'Training took {time.time()-start} seconds')
		self.plot_loss(train_losses, val_losses)

	def plot_loss(self, train_losses, val_losses, fig_title='loss.png'):
		x_vals = [x for x in range(self.epochs)]
		plt.plot(x_vals, train_losses, ["Training Loss"])
		plt.plot(x_vals, val_losses, ["Validation Loss"])
		plt.xlabel("Epochs")
		plt.ylabel("Cross-Entropy Loss")
		plt.title("Training and Validation Loss Curves")
		plt.savefig(fig_title)

	def test_model(self):
		start=time.time()
		context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
		response = self.model.generation(context, max_tokens=500)
		print(f'Inference took {time.time()-start} seconds')
		print("---")
		print(response)