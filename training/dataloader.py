from datasets import load_dataset
from torch.utils.data import Dataset

class TextData(Dataset):
    def __init__(self, datatset, tokenizer, block_size):
        super().__init__()
        
        