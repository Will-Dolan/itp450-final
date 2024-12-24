import random
import tiktoken
import torch
import datasets
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import multiprocessing
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.distributed as dist
import os

class Data:
	def __init__(self, batch_size, seq_size, num_samples=10000, distributed=False, rank=0):
		self.data_text = []		
		self.data = []
		self.train_data = []		
		self.val_data = []
		self.batch_size = batch_size
		self.seq_size = seq_size
		self.num_samples = num_samples
		self.distributed = distributed
		self.rank = rank

	def load_dataset(self):
		cores = multiprocessing.cpu_count()
		# Only print on main process if distributed
		if not self.distributed or self.rank == 0:
			print(f"Loading data with {cores//2} cores...")
		
		data_text = datasets.load_dataset('openwebtext', trust_remote_code=True, num_proc=cores//2)['train']
		# Use same seed across all processes for consistent shuffling
		self.data_text = data_text.shuffle(seed=42).select(range(int(self.num_samples)))

	def encode_data(self):
		self.tokenizer = tiktoken.get_encoding('gpt2')
		self.vocab_size = self.tokenizer.n_vocab
		self.encode = lambda s: self.tokenizer.encode(s)
		self.decode = lambda l: self.tokenizer.decode(l)
		
		if not self.distributed or self.rank == 0:
			print("Encoding data...")
		
		encoded_data = []
		for text in tqdm(self.data_text, disable=self.distributed and self.rank != 0):
			encoded = self.encode(text['text'])
			if len(encoded) <= self.seq_size:
				if self.rank == 0:
					print(f"Warning: Sequence of size {len(encoded)} needs padding!")
				context = encoded[0:-1]
				target = encoded[1:]
				encoded_data.append([
					torch.tensor(context, dtype=torch.long),
					torch.tensor(target, dtype=torch.long)
				])
			else:
				i = random.randint(0, len(encoded)-self.seq_size-1)
				context = torch.tensor(encoded[i:i+self.seq_size], dtype=torch.long)
				target = torch.tensor(encoded[i+1:i+self.seq_size+1], dtype=torch.long)
				encoded_data.append([context, target])
		
		self.data = encoded_data
		return self.vocab_size

	def split_dataset(self, val_split=0.2):
		# Convert data list to custom Dataset
		class TextDataset(Dataset):
			def __init__(self, data):
				self.data = data
			
			def __len__(self):
				return len(self.data)
			
			def __getitem__(self, idx):
				return self.data[idx][0], self.data[idx][1]

		# Split indices
		indices = list(range(len(self.data)))
		split_idx = int(len(indices) * (1 - val_split))
		train_indices = indices[:split_idx]
		val_indices = indices[split_idx:]

		full_dataset = TextDataset(self.data)
		self.train_data = torch.utils.data.Subset(full_dataset, train_indices)
		self.val_data = torch.utils.data.Subset(full_dataset, val_indices)

		if self.distributed:
			train_sampler = DistributedSampler(
				self.train_data,
				num_replicas=dist.get_world_size(),
				rank=self.rank,
				shuffle=True
			)
			val_sampler = DistributedSampler(
				self.val_data,
				num_replicas=dist.get_world_size(),
				rank=self.rank,
				shuffle=False
			)
		else:
			train_sampler = None
			val_sampler = None

		self.train_dataloader = DataLoader(
			self.train_data,
			batch_size=self.batch_size,
			shuffle=(train_sampler is None),
			sampler=train_sampler,
			num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
			pin_memory=True
		)
		
		self.val_dataloader = DataLoader(
			self.val_data,
			batch_size=self.batch_size,
			shuffle=False,
			sampler=val_sampler,
			num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
			pin_memory=True
		)