import torch

class Data:
    def __init__(self):
        # TODO: take in input to configure these
        self.train_data = []
        self.val_data = []
        self.seq_size = 8
        self.batch_size = 8
        self.data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

    def load_dataset(self):
        # TODO: load dataset from url or elsewhere
        pass

    def encode_data(self):
        # TODO: encode loaded dataset with tiktoken
        pass

    def split_data(self):
        # TODO: split data into train/val
        pass

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