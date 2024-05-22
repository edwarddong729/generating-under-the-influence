import random
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from torch.nn import functional as F
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('emanjavacas/MacBERTh', model_max_length=64)
random.seed(15324)
torch.manual_seed(15324)


def inner_critic_training_dataloaders(train_samples_per, val_samples_per, eval_samples_per, batch_size):

    train_datasets = []
    val_datasets = []
    eval_datasets = []
    
    # This ordering/mapping will be maintained throughout
    for ind, style in enumerate(['A', 'B', 'C', 'D', 'E']):
        chosen_sentences, remainder_sentences = split_n_sentences(f'../corpus/sents_{style}.txt', train_samples_per + val_samples_per + eval_samples_per)
        train_sentences = chosen_sentences[:train_samples_per]
        val_sentences = chosen_sentences[train_samples_per:train_samples_per+val_samples_per]
        eval_sentences = chosen_sentences[train_samples_per+val_samples_per:]
        
        train_datasets.append(ClassificationDataset(train_sentences, label=ind))
        val_datasets.append(ClassificationDataset(val_sentences, label=ind))
        eval_datasets.append(ClassificationDataset(eval_sentences, label=ind))
        
        non_train = val_sentences + eval_sentences + remainder_sentences
        save_sentences(f'../corpus/nontrain_sents_{style}.txt', non_train)

    combined_train = torch.utils.data.ConcatDataset(train_datasets)
    combined_val = torch.utils.data.ConcatDataset(val_datasets)
    combined_eval = torch.utils.data.ConcatDataset(eval_datasets)
    
    train_dataloader = DataLoader(combined_train, batch_size=batch_size, shuffle=True, collate_fn=classification_training_collate_no_corruption)
    val_dataloader = DataLoader(combined_val, batch_size=batch_size, shuffle=True, collate_fn=classification_training_collate_no_corruption)
    eval_dataloader = DataLoader(combined_eval, batch_size=batch_size, shuffle=True, collate_fn=classification_training_collate_no_corruption)

    return train_dataloader, val_dataloader, eval_dataloader


def adversarial_dataloaders(batch_size):
    
    adversarial_datasets = []
    
    for ind, author in enumerate(['achebe', 'baldwin', 'jackson', 'oconnor', 'salinger']):
        adversarial_sentences, _ = split_n_sentences(f'../corpus/nontrain_sents_{author}.txt', 1024, shuffle=False)
        adversarial_datasets.append(ClassificationDataset(adversarial_sentences, label=ind))

    combined_adversarial = torch.utils.data.ConcatDataset(adversarial_datasets)
    adversarial_dataloader = DataLoader(combined_adversarial, batch_size=batch_size, shuffle=True, collate_fn=classification_training_collate_no_corruption)

    return adversarial_dataloader


def gen_warmup_dataloaders(style, train_sample_num, val_sample_num, batch_size):

    chosen_sentences, _ = split_n_sentences(f'../corpus/sents_{style}.txt', train_sample_num + val_sample_num)

    train_sentences = chosen_sentences[:train_sample_num]
    val_sentences = chosen_sentences[train_sample_num:]

    train_dataset = CausalDataset(train_sentences)
    val_dataset = CausalDataset(val_sentences)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=causal_training_collate) if train_sample_num > 0 else None
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=causal_training_collate) if val_sample_num > 0 else None

    return train_dataloader, val_dataloader


def gen_warmup_all_styles_dataloader(train_samples_per, val_samples_per, batch_size):

    train_sentences = []
    val_sentences = []
    for author in ['achebe', 'baldwin', 'jackson', 'oconnor', 'salinger']:
        chosen_sentences, reserved_sentences = split_n_sentences(f'../corpus/sents_{author}.txt', train_samples_per + val_samples_per)
        train_sentences.extend(chosen_sentences[:train_samples_per])
        val_sentences.extend(chosen_sentences[train_samples_per:])
        save_sentences(f'../corpus/reserved_sents_{author}.txt', reserved_sentences)

    train_dataset = CausalDataset(train_sentences)
    val_dataset = CausalDataset(val_sentences)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=causal_training_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=causal_training_collate)

    return train_dataloader, val_dataloader


class ClassificationDataset(Dataset):

    def __init__(self, data, label):
        self.sentences = data
        self.labels = torch.full((len(self.sentences),), label, dtype=torch.long)
        encodings = [tokenizer(sentence, return_tensors='pt', truncation=True, padding=True) for sentence in self.sentences]
        self.input_ids = [encoding['input_ids'].squeeze() for encoding in encodings]
        self.attention_mask = [encoding['attention_mask'].squeeze() for encoding in encodings]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx], 'labels': self.labels[idx]}


class CausalDataset(Dataset):

    def __init__(self, data):
        self.sentences = data
        self.input_ids = [tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)['input_ids'].squeeze() for sentence in self.sentences]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.input_ids[idx]


def split_n_sentences(file_name, n, shuffle=True):
    with open(file_name, 'r') as file:
        # [CLS] and [SEP] added by default
        sentences = [line.strip() for line in file.readlines() if line]
    if shuffle:
        random.shuffle(sentences)

    # return tuple (sentences_to_use, sentences_to_ignore)
    return sentences[:n], sentences[n:]


def classification_training_collate_w_corruption(batch):

    max_len = max(len(item['input_ids']) for item in batch)

    padded_inputs = []
    attention_masks = []
    labels = []

    for item in batch:
        input_ids, attention_mask, label = probabilistic_corruption(item)

        # Pad input_ids and attention_mask to the maximum length
        padded_input_ids = F.pad(input_ids, (0, max_len - len(input_ids)), value=tokenizer.pad_token_id)
        padded_attention_mask = F.pad(attention_mask, (0, max_len - len(attention_mask)), value=0)

        padded_inputs.append(padded_input_ids)
        attention_masks.append(padded_attention_mask)
        labels.append(label)

    return {
        'input_ids': torch.stack(padded_inputs),
        'attention_mask': torch.stack(attention_masks),
        'labels': torch.tensor(labels)
    }


def classification_training_collate_no_corruption(batch):

    max_len = max(len(item['input_ids']) for item in batch)

    padded_inputs = []
    attention_masks = []
    labels = []

    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        label = item['labels']

        # Pad input_ids and attention_mask to the maximum length
        padded_input_ids = F.pad(input_ids, (0, max_len - len(input_ids)), value=tokenizer.pad_token_id)
        padded_attention_mask = F.pad(attention_mask, (0, max_len - len(attention_mask)), value=0)

        padded_inputs.append(padded_input_ids)
        attention_masks.append(padded_attention_mask)
        labels.append(label)

    return {
        'input_ids': torch.stack(padded_inputs),
        'attention_mask': torch.stack(attention_masks),
        'labels': torch.tensor(labels)
    }


def causal_training_collate(batch):

    max_len = max(len(sequence) for sequence in batch)

    padded_sequences = [torch.cat((sequence, torch.ones((max_len - len(sequence)), dtype=torch.long) * tokenizer.pad_token_id), dim=0) for sequence in batch]
    labels = [torch.cat((sequence[1:], torch.tensor([tokenizer.pad_token_id])), dim=0) for sequence in padded_sequences]
    loss_masks = torch.tensor([[1] * (len(sequence) - 1) + [0] * (max_len - len(sequence) + 1) for sequence in batch])

    # up to -1 to remove last position (where all sequences in batch have padding token as target)
    return torch.stack(padded_sequences)[:, :-1], torch.stack(labels)[:, :-1], loss_masks[:, :-1]


def probabilistic_corruption(batch_item):
    """ Corrupt an average of 25% of batch data by swapping token order and replacing tokens """

    input_ids = batch_item['input_ids']
    attention_mask = batch_item['attention_mask']
    label = batch_item['labels']

    if random.random() > 1/4:
        return input_ids, attention_mask, label
    
    permute_boundaries = random.sample(range(1, input_ids.size(0)), 2)
    section_to_permute = input_ids[min(permute_boundaries):max(permute_boundaries)]
    permuted_indices = torch.randperm(section_to_permute.size(0))
    section_to_permute = section_to_permute[permuted_indices]
    
    if section_to_permute.size(0) > 4:
        if random.random() > 1/2:
            replaced_position = random.randint(0, section_to_permute.size(0) - 1)
            replacement = random.randint(0, len(tokenizer) - 1)
            section_to_permute[replaced_position] = replacement
    
    input_ids = torch.cat([input_ids[:min(permute_boundaries)], section_to_permute, input_ids[max(permute_boundaries):]])

    return input_ids, attention_mask, torch.tensor(5)


def save_sentences(file_path: str, sentences: list):
    """ Save list of sentences to path """
    
    with open(file_path, 'w') as file:
        for sentence in sentences:
            file.write(sentence + '\n')
