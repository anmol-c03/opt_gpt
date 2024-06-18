import torch
import tiktoken
# from gpt2_0 import device


device='cpu'
#i am using cpu hence i am setting device to cpu 

class Dataloader:
    def __init__(self,B,T,filename=None):
        self.B = B
        self.T = T
        if filename is None:
            filename = 'input.txt'
        text=open(filename,'r').read() 
        enc=tiktoken.get_encoding('gpt2')
        self.tokens=torch.tensor(enc.encode(text))
        print(f'total number of tokens {len(self.tokens)}')
        print(f'1 epoch {len(self.tokens) // (B*T)} batches')

        self.current_pos=0
    
    def next_batch(self):
        if self.current_pos+(self.B*self.T)+1 >= len(self.tokens):
            self.current_pos=0
        batch=self.tokens[self.current_pos:self.current_pos+self.B*self.T+1]
        batch=batch.to(device)
        x=batch[:-1].view(self.B,self.T)
        y=batch[1:].view(self.B,self.T)
        self.current_pos+=self.B*self.T
        return x,y
