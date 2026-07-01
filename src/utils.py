import json
import yaml
from torch.utils.data import Dataset


def load_config(config_path: str) -> dict:
    """Load YAML config file and return as a dictionary."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def format_prompt(question: str, answer: str) -> str:
    """
    Format a Q&A pair into a single string prompt.
    
    Why? GPT-2 is a text completion model — it doesn't have
    separate 'question' and 'answer' fields. We concatenate them
    into one string with clear markers so the model learns the
    pattern: when it sees '### Answer:' after a question, it
    should complete with the right answer.
    """
    return f"### Question: {question}\n### Answer: {answer}"


class QADataset(Dataset):
    """
    PyTorch Dataset for our Q&A JSONL file.
    
    Why inherit from Dataset? PyTorch's DataLoader expects this
    interface — it needs __len__ and __getitem__ to batch and
    shuffle our data during training.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Read each line of the JSONL file as a separate example
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:  # skip empty lines
                    item = json.loads(line)
                    prompt = format_prompt(item["question"], item["answer"])
                    self.samples.append(prompt)

    def __len__(self):
        """How many examples do we have total?"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return one tokenized example at index idx.
        
        Tokenization converts the text string into a tensor of
        integers (token IDs) that the model can actually process.
        
        - padding='max_length': pad short sequences to max_length
        - truncation=True: cut sequences longer than max_length
        - return_tensors='pt': return PyTorch tensors not lists
        """
        encoding = self.tokenizer(
            self.samples[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()      # shape: (max_length,)
        attention_mask = encoding["attention_mask"].squeeze()  # shape: (max_length,)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),  # for causal LM, labels = input_ids
        }