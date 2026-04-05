from training.config import Config
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
import torch
from src.tokenizer import Tokenizer

config = Config()
tkn = Tokenizer(tokenizer_model=config.tkn_model)

INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}"

def format_and_tokenize(sample):
    
    instruction_text = INSTRUCTION_TEMPLATE.format(instruction=sample.get('instruction', ""))
    
    if sample.get("input", "").strip():
        instruction_text += f"{sample['input']}\n\n### Response:\n"
 
    response_text = sample.get("output", "")
    
    instruction_text_tokenized = tkn.encode(instruction_text)
    response_text_tokenized = tkn.encode(response_text)
    
    full = instruction_text_tokenized + response_text_tokenized + [tkn.eos_id]
    loss_mask = (
        [False] * len(instruction_text_tokenized) +
        [True]  * len(response_text_tokenized) +
        [True]
    )
    
    return full, loss_mask

class FineTuneDataset(IterableDataset):
    def __init__(self, epochs=config.epochs_ft, block_size=config.block_size, dataset=config.dataset_ft, split="train"):
        super().__init__()
        self.epochs = epochs
        self.block_size = block_size
        self.dataset = dataset
        self.split = split
        
    def __iter__(self):
        token_buf = []
        mask_buf  = []
 
        def flush():
            chunk = token_buf[:self.block_size + 1]
            cmask = mask_buf [:self.block_size + 1]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:],  dtype=torch.long)
            m = torch.tensor(cmask[1:],  dtype=torch.bool)
            return x, y, m
 
        for _ in range(self.epochs):
            dataset = load_dataset(self.dataset, split="train", streaming=True)
 
            for sample in dataset:
                ids, mask = format_and_tokenize(sample)
 
                if len(ids) > self.block_size + 1:
                    instr_len = sum(1 for m in mask if not m)
                    max_resp  = self.block_size + 1 - instr_len - 1
                    if max_resp < 1:
                        continue
                    ids  = ids [:instr_len + max_resp] + [tkn.eos_id]
                    mask = mask[:instr_len + max_resp] + [True]
 
                token_buf.extend(ids)
                mask_buf.extend(mask)
 
                while len(token_buf) >= self.block_size + 1:
                    yield flush()
                    token_buf = token_buf[self.block_size:]
                    mask_buf  = mask_buf [self.block_size:]
 
        if token_buf:
            pad = (self.block_size + 1) - len(token_buf)
            token_buf.extend([tkn.eos_id] * pad)
            mask_buf.extend ([False]      * pad)
            yield flush()
                
def get_data():
    dataset = FineTuneDataset()
    return DataLoader(dataset, batch_size=config.batch_size_ft, num_workers=config.num_workers, pin_memory=config.pin_memory)