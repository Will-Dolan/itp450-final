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
		self.embed_dim = 768
		self.n_heads = 12
		self.n_layers = 12
		self.learning_rate = 1e-4

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

	def train_model(self, save_model=True, model_path="model.pth"):
		torch.cuda.empty_cache()

		# Finetuning parameters to improve training loss
		print_interval=1 # print after every epoch
		eval_iters=200 # not used

		if torch.cuda.is_available():
			self.device = 'cuda'
		else: 
			self.device ='cpu'
		print('Running on', self.device)

		# Instantiating the Transformer module
		self.model = Transformer(self.embed_dim, self.n_heads, self.vocab_size, self.seq_size, self.n_layers, self.device)

		self.model = self.model.to(self.device)

		# Print the number of parameters in the model
		print(f"Num trainable params = {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
		print('Parameters size:', len(list(self.model.parameters())))

		# Optimizer
		optimizer = torch.optim.AdamW(self.model.parameters(),lr=self.learning_rate)
		curr_epochs = []
		train_losses = []
		val_losses = []

		# Training Loop
		start=time.time()
		for epoch in range(self.epochs):
			self.model.train()
			optimizer.zero_grad()
			print("Epoch " + str(epoch))
			train_loss = 0
			for batch, (context, target) in tqdm(enumerate(self.dataset.train_dataloader)):
				context.to(self.device)
				target.to(self.device)
				_, loss = self.model(context, target)
				loss.backward()
				optimizer.step()
				train_loss += loss.item()
			train_loss /= (batch+1)
			train_losses.append(train_loss)
   
			with torch.no_grad():
				val_loss = 0
				for val_batch, (val_context, val_target) in tqdm(enumerate(self.dataset.val_dataloader)):
					val_context = val_context.to(self.device)
					val_target = val_target.to(self.device)
					_, loss = self.model(val_context, val_target)
					val_loss += loss.item()
				val_loss /= (val_batch+1)
				val_losses.append(val_loss)
    
			if epoch % print_interval == 0 or epoch == self.epochs - 1 or epoch == 0:
				print(f"[{(time.time()-start):.2f}s] step {epoch}: train loss {train_loss}, val loss {val_loss}")
				self.plot_loss(train_losses, val_losses, epoch)
    
		print(f'Training took {time.time()-start} seconds')
		if save_model: torch.save(self.model.state_dict(), model_path)

	def plot_loss(self, train_losses, val_losses, epoch, fig_title='loss.png'):
		x_vals = [x for x in range(epoch+1)]
		plt.plot(x_vals, train_losses, ["Training Loss"])
		plt.plot(x_vals, val_losses, ["Validation Loss"])
		plt.xlabel("Epochs")
		plt.ylabel("Cross-Entropy Loss")
		plt.title("Training and Validation Loss Curves")
		plt.savefig(fig_title)

	def test_model(self, load_model=False, model_path="model.pth"):
		if load_model: 
			self.model = Transformer(self.embed_dim, self.n_heads, self.vocab_size, self.seq_size, self.n_layers, self.device)
			self.model.load_state_dict(torch.load(model_path, weights_only=True))
		start=time.time()
		# context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
		sample_idx = 0
		context = self.dataset.val_data[sample_idx][0]
		response = self.model.generation(context, max_tokens=500)
		print(f'Inference took {time.time()-start} seconds')
		print("---")
		print(response)