from datasets import load_dataset

dataset = load_dataset(
    "draco976/wikipedia-bookcorpus",
    split="train"
)

print(dataset)