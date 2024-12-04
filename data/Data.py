import tiktoken
import torch

class Data:
	def __init__(self, batch_size, seq_size):
		# TODO: take in input to configure these
		self.train_data = []
		self.val_data = []
		self.seq_size = seq_size
		self.batch_size = batch_size
		self.data_path = 'input.txt'

	def load_dataset(self):
		# path is from TLD, where we run main
		with open('data/input.txt', 'r', encoding='utf-8') as f:
			self.text = f.read()

	def encode_data(self):
		self.tokenizer = tiktoken.get_encoding('gpt2')
		self.vocab_size = self.tokenizer.n_vocab
		# Set up encoding and decoding
		self.encode = lambda s: self.tokenizer.encode(s)
		self.decode = lambda l: self.tokenizer.decode(l)
		# Encode the text and store it as a tensor
		self.data = torch.tensor(self.encode(self.text.strip()), dtype=torch.long)

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

	def split_data(self):
		n = int(0.9*len(self.data))
		self.train_data = self.data[:n]
		self.val_data   = self.data[n:]

	def get_batch(self, split):
		data = self.train_data if split=='train' else self.val_data

		# Randomly select starting indices for the batch (0, len(data)-seq_size)
		# Do it for batch_size number of times and store in a torch vector
		start_ind = torch.randint(0,len(data)-self.seq_size,(self.batch_size,))
		#print(f'start_ind={start_ind}')
		
		context=[]
		target=[]
		for i in start_ind:
			context.append(data[i  :i+self.seq_size  ])
			target.append (data[i+1:i+self.seq_size+1])
		
		context = torch.stack(context) #converts from a list of tensors to a large tensor
		target  = torch.stack(target)

		return context, target