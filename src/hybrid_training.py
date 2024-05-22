import numpy as np
import sys
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from custom_models import CustomGenerator
from dataloaders import gen_warmup_dataloaders  


def masked_ce_loss(logits, targets, masks):
    
    # Compute the standard cross-entropy loss
    ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
    # Apply the mask to the loss
    masked_loss = ce_loss * masks
    # Compute the total loss by summing over non-padding tokens
    total_loss = masked_loss.sum()
    # Normalize the loss by the number of non-padding tokens
    num_non_padding_tokens = masks.sum()
    return total_loss / num_non_padding_tokens


def pad_out_tokens_after_eos(batch_sequences, eos_token_id, pad_token_id):
    
    eos_indices = (batch_sequences == eos_token_id).nonzero()
    # Initialize a mask tensor for each sequence in the batch
    masks = torch.ones_like(batch_sequences)
    for eos_index in eos_indices:
        seq_index, token_index = eos_index.tolist()
        masks[seq_index, token_index + 1:] = pad_token_id
    replaced_batch_sequences = torch.where(masks == pad_token_id, pad_token_id, batch_sequences)
    return replaced_batch_sequences


def build_models(device):

    # LOAD DISCRIMINATOR
    discriminator = BertForSequenceClassification.from_pretrained('../models/warmed_up_models/best_disc_04_a')

    for _, param in discriminator.named_parameters():
        param.requires_grad = False

    # LOAD GENERATOR
    tokenizer = BertTokenizer.from_pretrained('emanjavacas/MacBERTh', model_max_length=64)
    vocab_size = len(tokenizer)
    generator = CustomGenerator(vocab_size)

    for name, param in generator.named_parameters():
        if name.startswith('macberth.embeddings.'):
            param.requires_grad = False
        elif name.startswith('macberth.encoder.layer.0.'):
            param.requires_grad = False
        elif name.startswith('macberth.encoder.layer.1.'):
            param.requires_grad = False
        elif name.startswith('macberth.encoder.layer.2.'):
            param.requires_grad = False
        elif name.startswith('macberth.encoder.layer.3.'):
            param.requires_grad = False
        elif name.startswith('macberth.encoder.layer.4.'):
            param.requires_grad = False
        elif name.startswith('macberth.encoder.layer.5.'):
            param.requires_grad = False
        elif name.startswith('macberth.encoder.layer.6.'):
            param.requires_grad = False
        elif name.startswith('macberth.encoder.layer.7.'):
            param.requires_grad = False
        else:
            param.requires_grad = True

    return tokenizer, generator, discriminator


def main():

    INIT_STYLE, INFL_STYLE = sys.argv[1]
    MAX_EPOCHS = 35  # 65 # 50
    TRAINING_SAMPLES = 6020  # 3816  # 3808  # 3823
    LR = 3e-5  #6e-6  # 5e-6
    BATCH_SIZE = 20  # 24  # 16  # 32
    ADV_LOSS_SCALAR = int(sys.argv[2])  # try 0, 25, 50, 100, 200
    # ADV_TRAINING_REPS = 0

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {DEVICE}')
    torch.manual_seed(15324)

    dataloader, _ = gen_warmup_dataloaders(INIT_STYLE, TRAINING_SAMPLES, 0, BATCH_SIZE)
    print(f'Batch count: {len(dataloader)}')

    tokenizer, generator, discriminator = build_models(DEVICE)
    generator.to(DEVICE)
    discriminator.to(DEVICE)
    
    generator.train()
    discriminator.eval()

    embedding_weights = discriminator.get_input_embeddings().weight
    print('Warm discriminator loaded, cold generator initialized')

    opt = torch.optim.AdamW(generator.parameters(), lr=LR)

    styles_mapper = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4
    }
    # a little pressure to evolve from push perspective, mostly should come from pull
    weights_for_adv_loss = torch.zeros(5).to(DEVICE)
    weights_for_adv_loss[styles_mapper[INIT_STYLE]] = 1
    weights_for_adv_loss[styles_mapper[INFL_STYLE]] = 1
    
    target = torch.ones(5).to(DEVICE) * 0.2
    target[styles_mapper[INIT_STYLE]] = 1
    target[styles_mapper[INFL_STYLE]] = 1
    
    print(f'Weights for gen loss function: {weights_for_adv_loss}')

    # applies specifically to computation of loss for adversarial updates
    criterion_adv = torch.nn.CrossEntropyLoss(weights_for_adv_loss)

    # remember to toggle between model.train and model.eval, and to use context handler
    losses_autoregressive = []
    losses_adversarial = []

    for epoch in trange(MAX_EPOCHS):
        
        print(f'---------------------- Starting EPOCH {epoch} ----------------------')

        for batch_num, (batch_inputs, batch_labels, loss_masks) in enumerate(dataloader):
            # print(f'++++++++++++++ batch {batch_num} ++++++++++++++')

            opt.zero_grad()

            # AUTOREGRESSIVE TRAINING
            batch_inputs = batch_inputs.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            batch_loss_masks = loss_masks.to(DEVICE)

            loss_autoregressive = 0.
            for i in range(batch_inputs.size(1)):
                
                # teacher forcing
                step_inputs = batch_inputs[:, :i + 1]
                step_labels = batch_labels[:, i]
                step_masks = batch_loss_masks[:, i]

                # Forward pass
                logits, _ = generator(step_inputs)
                loss_autoregressive += masked_ce_loss(logits, step_labels, step_masks)

            if epoch < 5:
                loss_autoregressive.backward()
                opt.step()
                continue

            # ADVERSARIAL TRAINING
            # for _ in range(ADV_TRAINING_REPS):
            index_sequence = torch.tensor([[tokenizer.cls_token_id]] * BATCH_SIZE).to(DEVICE)
            one_hot_sequence = torch.zeros(BATCH_SIZE, 1, len(tokenizer)).to(DEVICE)
            one_hot_sequence[:, 0, tokenizer.cls_token_id] = 1
            
            pad_out = torch.ones(BATCH_SIZE, 64).to(DEVICE)
            sequence_lengths = torch.ones(BATCH_SIZE, dtype=torch.short).to(DEVICE) * 64
            for token_ind in range(64 - 1):
                _, one_hots = generator(index_sequence, requires_gumbel_out=True)
                next_token_id = torch.argmax(one_hots, dim=1).unsqueeze(dim=1)
                index_sequence = torch.cat([index_sequence, next_token_id], dim=1)
                one_hot_sequence = torch.cat([one_hot_sequence, one_hots.unsqueeze(1)], dim=1)

                for seq_ind in range(BATCH_SIZE):
                    if next_token_id[seq_ind].item() == tokenizer.sep_token_id:
                        pad_out[seq_ind, token_ind + 2:] = 0
                        sequence_lengths[seq_ind] = token_ind + 2
            
            index_sequence = pad_out_tokens_after_eos(index_sequence, tokenizer.sep_token_id, tokenizer.pad_token_id)
            
            if batch_num % 30 == 0:
                for ind in range(BATCH_SIZE):
                    print(tokenizer.decode(index_sequence[ind], skip_special_tokens=True))

            padded_out_one_hots = one_hot_sequence * pad_out.unsqueeze(-1)
            assert padded_out_one_hots.requires_grad
            # padded_sequences = pad_sequence(padded_sequences, batch_first=True)  # still necessary?
            embedded_sequences = torch.matmul(padded_out_one_hots, embedding_weights)

            attention_mask = torch.ones(padded_out_one_hots.size(0), padded_out_one_hots.size(1), dtype=torch.bool).to(DEVICE)
            for sequence_num, length in enumerate(sequence_lengths):
                attention_mask[sequence_num, length:] = 0

            stacked_labels = target.repeat(BATCH_SIZE, 1)
            judgment_logits = discriminator(inputs_embeds=embedded_sequences, attention_mask=attention_mask, return_dict=False)[0]
            loss_adversarial = criterion_adv(judgment_logits, stacked_labels)

            total_loss = loss_autoregressive + loss_adversarial * ADV_LOSS_SCALAR
            total_loss.backward()
            opt.step()

            losses_adversarial.append(loss_adversarial.item() * ADV_LOSS_SCALAR)
            losses_autoregressive.append(loss_autoregressive.item())
            print(f'ADV Loss (scaled): {loss_adversarial.item() * ADV_LOSS_SCALAR:.2f}\tAUT Loss: {loss_autoregressive.item():.2f}\n')

    generator.eval()
    print('FINAL sample outputs:')
    with torch.no_grad():
        index_sequence = torch.tensor([[tokenizer.cls_token_id]] * BATCH_SIZE).to(DEVICE)
        for _ in range(64 - 1):
            _, one_hots = generator(index_sequence, requires_gumbel_out=True)
            next_token_id = torch.argmax(one_hots, dim=1).unsqueeze(dim=1)
            index_sequence = torch.cat([index_sequence, next_token_id], dim=1)
        index_sequence = pad_out_tokens_after_eos(index_sequence, tokenizer.sep_token_id, tokenizer.pad_token_id)
        for ind in range(BATCH_SIZE):
            print(tokenizer.decode(index_sequence[ind], skip_special_tokens=True))
    
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.plot(range(len(losses_adversarial)), losses_adversarial)
    plt.plot(range(len(losses_autoregressive)), losses_autoregressive)
    plt.legend(('adversarial loss', 'autoregressive loss'))
    plt.savefig(f'loss_trends_{INIT_STYLE}{INFL_STYLE}{ADV_LOSS_SCALAR}.png')
    plt.clf()


if __name__=='__main__':
    main()
