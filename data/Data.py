import random
import tiktoken
import torch
import datasets
from torch.utils.data import Dataset, DataLoader
import datasets
import multiprocessing
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class Data:
	def __init__(self, batch_size, seq_size, num_samples=10000):
		self.data_text = []		
		self.data = []
		self.train_data = []		
		self.val_data = []
		self.batch_size = batch_size
		self.seq_size = seq_size
		self.num_samples = num_samples

	def load_dataset(self):
		cores = multiprocessing.cpu_count()
		# Load openwebtext dataset
		print("Loading data...")
		data_text = datasets.load_dataset('openwebtext', trust_remote_code=True, num_proc=cores//2)['train']
		self.data_text = data_text.shuffle(seed=42).select(range(int(self.num_samples)))

	def encode_data(self):
		self.tokenizer = tiktoken.get_encoding('gpt2')
		self.vocab_size = self.tokenizer.n_vocab
		# Set up encoding and decoding
		self.encode = lambda s: self.tokenizer.encode(s)
		self.decode = lambda l: self.tokenizer.decode(l)
		# Encode the text and store it as a tensor
		print("Encoding data...")
		for text in tqdm(self.data_text):
			encoded = self.encode(text['text'])
			if(len(encoded) <= self.seq_size): # TODO: add padding (optional)
				print("Need to add padding on sequence of size " + str(len(encoded)) + "!")
				context = encoded[0:-1]
				target = encoded[1:]
				self.data.append([context,target])
			else:
				i = random.randint(0,len(encoded)-self.seq_size-1)
				context = torch.tensor(encoded[i:i+self.seq_size], dtype=torch.long)
				target = torch.tensor(encoded[i+1:i+self.seq_size+1], dtype=torch.long)
				self.data.append([context,target])

		return self.vocab_size

		# chars = sorted(set(self.text))
		# self.vocab_size = len(chars)
		#
		# c2i = {c: i for i, c in enumerate(chars)}  # character to integer
		# i2c = {i: c for i, c in enumerate(chars)}  # integer to character
		#
		# # encode a string to a "list of integers"
		# self.encode = lambda s: [c2i[c] for c in s]
		#
		# # decode a list of integers to a string
		# self.decode = lambda l: ''.join([i2c[i] for i in l])
		#
		# self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
		#
		# return self.vocab_size

	def split_dataset(self, val_split=0.2):
		self.val_data = self.data[:int(self.num_samples*val_split)]
		self.train_data = self.data[int(self.num_samples*val_split):]
		self.val_dataloader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=True)
		self.train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

	# def get_batch(self, split):
	# 	data = self.train_data_enc if split=='train' else self.val_data_enc

	# 	# Randomly select starting indices for the batch (0, len(data)-seq_size)
	# 	# Do it for batch_size number of times and store in a torch vector
	# 	start_ind = torch.randint(0,len(data)-self.seq_size,(self.batch_size,))
	# 	#print(f'start_ind={start_ind}')
		
	# 	context=[]
	# 	target=[]
	# 	for i in start_ind:
	# 		context.append(data[i  :i+self.seq_size  ])
	# 		target.append (data[i+1:i+self.seq_size+1])
		
	# 	context = torch.stack(context) #converts from a list of tensors to a large tensor
	# 	target  = torch.stack(target)

	# 	return context, target