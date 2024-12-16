import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.Transformer import Transformer
from tqdm import tqdm
import os
from data.Data import Data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class Pipeline:
	def __init__(self, args):
		print("Initializing pipeline")
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
		self.n_heads = 32
		self.n_layers = 32
		self.learning_rate = 1e-4

		self.distributed = False

		# Ensure reproducibility
		if self.seed is not None:
			torch.manual_seed(self.seed)
			np.random.seed(self.seed)

	def configure_gpus(self):

		if torch.cuda.is_available():
			self.device = 'cuda'
		else: 
			self.device = 'cpu'
		print('Running on', self.device)
		
		# Only configure distributed if SLURM environment variables are present
		if all(env in os.environ for env in ["SLURM_PROCID", "WORLD_SIZE", "SLURM_GPUS_ON_NODE"]):
			self.distributed = True

			rank		  = int(os.environ["SLURM_PROCID"])
			world_size	= int(os.environ["WORLD_SIZE"])
			gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
			assert gpus_per_node == torch.cuda.device_count()

			print(f"Hello from rank {rank} of {world_size}  where there are" \
				f" {gpus_per_node} allocated GPUs per node.", flush=True)

			dist.init_process_group("nccl", rank=rank, world_size=world_size)
			if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

			self.local_rank = rank - gpus_per_node * (rank // gpus_per_node)
			torch.cuda.set_device(self.local_rank)
			self.device = self.local_rank # for clarity with train_model
			self.rank = rank

	def load_dataset(self):
		val_split = 0.2
		start=time.time()
		self.dataset = Data(self.batch_size, self.seq_size, num_samples=self.num_samples)
		self.dataset.load_dataset()
		self.vocab_size = self.dataset.encode_data()
		self.dataset.split_dataset(val_split)
		print(f'Data loading took {time.time()-start} seconds')

	def load_model(self):
		self.model = Transformer(self.embed_dim, self.n_heads, self.vocab_size, 
							   self.seq_size, self.n_layers, self.device)
		self.model = self.model.to(self.device)
		
		if self.distributed:
			# Wrap model in DistributedDataParallel

			self.model = DDP(self.model, device_ids=[self.local_rank])
		
		if self.rank == 0:  # Only print on main process
			print(f"Num trainable params = {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
			print('Parameters size:', len(list(self.model.parameters())))

		self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
		

	def train_model(self, save_model=True, model_path="model.pth"):
		torch.cuda.empty_cache()

		# Finetuning parameters to improve training loss
		print_interval=1 # print after every epoch
		eval_iters=200 # not used

		curr_epochs = []
		train_losses = []
		val_losses = []

		# Training Loop
		start=time.time()
		for epoch in range(self.epochs):
			self.model.train()
			self.optimizer.zero_grad()
			print("Epoch " + str(epoch))
			train_loss = 0
			for batch, (context, target) in tqdm(enumerate(self.dataset.train_dataloader)):
				context.to(self.device)
				target.to(self.device)
				_, loss = self.model(context, target)
				loss.backward()
				self.optimizer.step()
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
				with open("train_losses.txt", "w") as file: 
					for loss in train_losses: file.write(str(loss) + ' ')
					file.write('\n')
				with open("val_losses.txt", "w") as file:
					for loss in val_losses: file.write(str(loss) + ' ')
					file.write('\n')
		
		print(f'Training took {time.time()-start} seconds')
		if save_model: torch.save(self.model.state_dict(), model_path)

	def plot_loss(self, fig_title='loss.png'):
		x_vals = [x for x in range(self.epochs)]
		train_losses = np.loadtxt('train_losses.txt', delimiter=' ')
		val_losses = np.loadtxt('val_losses.txt', delimiter=' ')
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
		sample_idx = 1
		context = self.dataset.val_data[sample_idx][0][np.newaxis, :]
		response = self.model.generation(context, max_tokens=500)
		print(f'Inference took {time.time()-start} seconds')
		print("---")
		print(response)

	def cleanup(self):
		if self.distributed and dist.is_initialized():
			dist.destroy_process_group()
			print("Cleaned up distributed process group")