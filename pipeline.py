import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.Transformer import Transformer
import os
from data.Data import Data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class Pipeline:
	def __init__(self, args):
		if dist.is_available() and dist.is_initialized():
			self.rank = dist.get_rank()
		else:
			self.rank = 0
			
		if self.rank == 0:
			print("Initializing pipeline")

		# defaults are configured in main.py
		self.args = args
		self.seq_size = args.seq_size
		self.batch_size = args.batch_size 
		self.epochs = args.epochs
		self.saved_model_pathway = args.saved_model_pathway
		self.seed = args.seed
		self.num_samples = args.num_samples
		self.embed_dim = 768
		self.n_heads = args.num_heads
		self.n_layers = args.num_layers
		self.learning_rate = 1e-4
		self.distributed = False

		# Ensure reproducibility
		if self.seed is not None:
			torch.manual_seed(self.seed)
			np.random.seed(self.seed)
			torch.cuda.manual_seed_all(self.seed)
			torch.backends.cudnn.deterministic = True

	def configure_gpus(self):
		if torch.cuda.is_available():
			self.device = 'cuda'
		else: 
			self.device = 'cpu'
			
		if self.rank == 0:
			print('Running on', self.device, f'with {torch.cuda.device_count()} ')
		
		# Only configure distributed if SLURM environment variables are present
		if all(env in os.environ for env in ["SLURM_PROCID", "WORLD_SIZE", "SLURM_GPUS_ON_NODE"]):
			self.distributed = True

			rank = int(os.environ["SLURM_PROCID"])
			world_size = int(os.environ["WORLD_SIZE"])
			gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
			assert gpus_per_node == torch.cuda.device_count(), \
				f'Requested {gpus_per_node} GPUs but only got {torch.cuda.device_count()}'

			if rank == 0:
				print(f"Setting up distributed training with {world_size} processes")
				print(f"GPUs per node: {gpus_per_node}")

			# Initialize process group if not already initialized
			if not dist.is_initialized():
				dist.init_process_group("nccl", rank=rank, world_size=world_size)
				if rank == 0:
					print(f"Process group initialized: {dist.is_initialized()}")

			self.local_rank = rank - gpus_per_node * (rank // gpus_per_node)
			torch.cuda.set_device(self.local_rank)
			self.device = self.local_rank
			self.rank = rank

			# Set this for proper distributed training
			self.batch_size = self.batch_size // world_size

	def load_dataset(self):
		val_split = 0.2
		start = time.time()
		self.dataset = Data(
			self.batch_size, 
			self.seq_size, 
			num_samples=self.num_samples,
			distributed=self.distributed,
			rank=self.rank if self.distributed else 0
		)
		self.dataset.load_dataset()
		self.vocab_size = self.dataset.encode_data()
		self.dataset.split_dataset(val_split)
		
		if self.rank == 0:
			print(f'Data loading took {time.time()-start} seconds')

	def load_model(self):
		self.model = Transformer(
			self.embed_dim, 
			self.n_heads, 
			self.vocab_size, 
			self.seq_size, 
			self.n_layers, 
			self.device
		)
		self.model = self.model.to(self.device)
		
		if self.distributed:
			self.model = DDP(
				self.model, 
				device_ids=[self.local_rank],
				output_device=self.local_rank,
				find_unused_parameters=False
			)
		
		if self.rank == 0:
			print(f"Num trainable params = {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
			print('Parameters size:', len(list(self.model.parameters())))

		self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

	def train_model(self, save_model=True, model_path="model.pth"):
		torch.cuda.empty_cache()

		print_interval=1 # print after every epoch
		train_losses = []
		val_losses = []

		# Training Loop
		start=time.time()
		for epoch in range(self.epochs):
			self.model.train()
			self.optimizer.zero_grad()
			train_loss = 0
			for batch, (context, target) in enumerate(self.dataset.train_dataloader):
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
				for val_batch, (val_context, val_target) in enumerate(self.dataset.val_dataloader):
					val_context = val_context.to(self.device)
					val_target = val_target.to(self.device)
					_, loss = self.model(val_context, val_target)
					val_loss += loss.item()
				val_loss /= (val_batch+1)
				val_losses.append(val_loss)
		
			if epoch % print_interval == 0 or epoch == self.epochs - 1 or epoch == 0:
				if not self.distributed or self.rank == 0:
					print(f"[{(time.time()-start):.2f}s] step {epoch}: train loss {train_loss}, val loss {val_loss}")
				with open("train_losses.txt", "w") as file: 
					for loss in train_losses: file.write(str(loss) + ' ')
					file.write('\n')
				with open("val_losses.txt", "w") as file:
					for loss in val_losses: file.write(str(loss) + ' ')
					file.write('\n')
		
		if self.rank == 0:
			print(f'Training took {time.time()-start} seconds')
			if save_model:
				# Save model on rank 0 only
				if isinstance(self.model, DDP):
					torch.save(self.model.module.state_dict(), model_path)
				else:
					torch.save(self.model.state_dict(), model_path)

	def plot_loss(self, fig_title='loss.png'):
		if self.rank == 0:  # Only plot on main process
			x_vals = list(range(self.epochs))
			train_losses = np.loadtxt('train_losses.txt', delimiter=' ')
			val_losses = np.loadtxt('val_losses.txt', delimiter=' ')
			
			plt.figure(figsize=(10, 6))
			plt.plot(x_vals, train_losses, label="Training Loss")
			plt.plot(x_vals, val_losses, label="Validation Loss")
			plt.xlabel("Epochs")
			plt.ylabel("Cross-Entropy Loss")
			plt.title("Training and Validation Loss Curves")
			plt.legend()
			plt.savefig(fig_title)
			plt.close()

	def test_model(self, load_model=False, model_path="model.pth"):
		if load_model:
			self.model = Transformer(
				self.embed_dim, 
				self.n_heads, 
				self.vocab_size, 
				self.seq_size, 
				self.n_layers, 
				self.device
			)
			# Load model on appropriate device
			state_dict = torch.load(model_path, map_location=f'cuda:{self.local_rank}' if self.distributed else self.device)
			self.model.load_state_dict(state_dict)
			self.model = self.model.to(self.device)
			
			if self.distributed:
				self.model = DDP(self.model, device_ids=[self.local_rank])

		start = time.time()
		sample_idx = 1
		context = self.dataset.val_data[sample_idx][0].unsqueeze(0).to(self.device)
		
		if isinstance(self.model, DDP):
			response = self.model.module.generation(context, max_tokens=500)
		else:
			response = self.model.generation(context, max_tokens=500)

		if not self.distributed or self.rank == 0:
			print(f'Inference took {time.time()-start} seconds')
			print("---")
			print(response)

	def cleanup(self):
		if self.distributed and dist.is_initialized():
			dist.destroy_process_group()
			if self.rank == 0:
				print("Cleaned up distributed process group")