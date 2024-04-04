import numpy as np
import sys
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from custom_models import CustomGenerator
from dataloaders import adversarial_dataloaders


def pad_out_tokens_after_eos(batch_sequences, eos_token_id, pad_token_id):
    
    eos_indices = (batch_sequences == eos_token_id).nonzero()
    # Initialize a mask tensor for each sequence in the batch
    masks = torch.ones_like(batch_sequences)
    for eos_index in eos_indices:
        seq_index, token_index = eos_index.tolist()
        masks[seq_index, token_index + 1:] = pad_token_id
    replaced_batch_sequences = torch.where(masks == pad_token_id, pad_token_id, batch_sequences)
    return replaced_batch_sequences


def load_warm_models(device, generator_style_id):

    # LOAD DISCRIMINATOR
    discriminator = BertForSequenceClassification.from_pretrained('../models/warmed_up_models/discriminator')

    for name, param in discriminator.named_parameters():
        if name.startswith('bert.pooler.'):
            param.requires_grad = True
        elif name.startswith('classifier.'):
            param.requires_grad = True
        else: 
            param.requires_grad = False

    # LOAD GENERATOR
    tokenizer = BertTokenizer.from_pretrained('emanjavacas/MacBERTh', model_max_length=64)
    vocab_size = len(tokenizer)
    generator = CustomGenerator(vocab_size)
    state_dict = torch.load(f'../models/warmed_up_models/generators/{generator_style_id}_state_dict.pth', map_location=device)
    generator.load_state_dict(state_dict)
    print(f'Loaded {generator_style_id}_state_dict.pth')

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
    MAX_EPOCHS = 5
    LR_GEN = 1e-4
    LR_DISC = 2e-9
    BATCH_SIZE = 32

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {DEVICE}')
    torch.manual_seed(15324)

    tokenizer, generator, discriminator = load_warm_models(DEVICE, INIT_STYLE)
    generator.to(DEVICE)
    discriminator.to(DEVICE)
    discriminator.train()
    embedding_weights = discriminator.get_input_embeddings().weight
    print('Warmed-up models loaded!')

    opt_gen = torch.optim.AdamW(generator.parameters(), lr=LR_GEN)
    opt_disc = torch.optim.AdamW(discriminator.parameters(), lr=LR_DISC)

    dataloader = adversarial_dataloaders(BATCH_SIZE)
    print(len(dataloader))

    styles_mapper = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4
    }
    # a little pressure to evolve from push perspective, mostly should come from pull
    weights_for_gen_loss = torch.zeros(6).to(DEVICE)
    weights_for_gen_loss[styles_mapper[INIT_STYLE]] = 1
    weights_for_gen_loss[styles_mapper[INFL_STYLE]] = 1
    
    intensified_weights_for_gen_loss = torch.zeros(6).to(DEVICE)
    intensified_weights_for_gen_loss[styles_mapper[INIT_STYLE]] = 1
    intensified_weights_for_gen_loss[styles_mapper[INFL_STYLE]] = 1.25
    intensified_weights_for_gen_loss[5] = 1.25
    
    target_movements_dict = {}
    for ind, style in enumerate(['A', 'B', 'C', 'D', 'E', 'F']):
        if style == INIT_STYLE:
            if INIT_STYLE == INFL_STYLE:
                # intensified_weights_for_gen_loss[ind] = 1.5
                target_movements_dict[style] = np.ones(4800)
            else:
                target_movements_dict[style] = np.ones(4800)
                # intensified_weights_for_gen_loss[ind] = 1.5
                # target_movements_dict[style] = np.concatenate((np.linspace(0.9, 0.6, 601), np.ones(999) * 0.6))
        elif style == INFL_STYLE:
            # intensified_weights_for_gen_loss[ind] = 1.5
            # target_movements_dict[style] = np.concatenate((np.linspace(0.2, 0.8, 601), np.ones(999) * 0.8))
            target_movements_dict[style] = np.ones(4800)
        elif style == 'F':
            target_movements_dict[style] = np.zeros(4800)
        else:
            # target_movements_dict[style] = np.ones(1600) * 0.2
            target_movements_dict[style] = np.zeros(4800)

    moving_targets = torch.tensor(np.stack([
        target_movements_dict['A'], 
        target_movements_dict['B'], 
        target_movements_dict['C'], 
        target_movements_dict['D'], 
        target_movements_dict['E'], 
        target_movements_dict['F']
        ], axis=1)).to(DEVICE)
    # another option is to use both in each update, just average them
    # yet another option: use an array of target labels
    print(f'Early weights for gen loss function: {weights_for_gen_loss}')
    print(f'Late weights for gen loss function: {intensified_weights_for_gen_loss}')

    # applies specifically to computation of loss for GEN updates
    early_criterion = torch.nn.CrossEntropyLoss()
    late_criterion = torch.nn.CrossEntropyLoss()

    generator.train()

    # remember to toggle between model.train and model.eval, and to use context handler
    losses_gen = []
    avg_losses_disc = []
    real_losses_disc = []
    fake_losses_disc = []

    for epoch in range(MAX_EPOCHS):

        criterion = early_criterion if epoch < 6 else late_criterion
        
        print(f'---------------------- Starting EPOCH {epoch} ----------------------')

        for batch_num, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # for batch_num, batch in enumerate(dataloader):
            target = moving_targets[epoch * 160 + batch_num, :]
            # print(f'Target class distribution: {target}')
            
            # print(f'++++++++++++++ batch {batch_num} ++++++++++++++')

            opt_gen.zero_grad()

            index_sequence = torch.tensor([[tokenizer.cls_token_id]] * BATCH_SIZE).to(DEVICE)
            one_hot_sequence = torch.zeros(BATCH_SIZE, 1, len(tokenizer)).to(DEVICE)
            one_hot_sequence[:, 0, tokenizer.cls_token_id] = 1
            
            pad_out = torch.ones(BATCH_SIZE, 64).to(DEVICE)
            sequence_lengths = torch.ones(BATCH_SIZE, dtype=torch.short).to(DEVICE) * 64
            for token_ind in range(64 - 1):
                _, one_hots = generator(index_sequence, requires_gumbel_out=True)

                next_token_id = torch.argmax(one_hots, dim=1).unsqueeze(dim=1)
                index_sequence = torch.cat([index_sequence, next_token_id], dim=1)  #

                one_hot_sequence = torch.cat([one_hot_sequence, one_hots.unsqueeze(1)], dim=1)

                for seq_ind in range(BATCH_SIZE):
                    if next_token_id[seq_ind].item() == tokenizer.sep_token_id:
                        pad_out[seq_ind, token_ind + 2:] = 0
                        sequence_lengths[seq_ind] = token_ind + 2
            
            # print('INDEX SEQUENCE')
            # print(index_sequence)
            padded_out_one_hots = one_hot_sequence * pad_out.unsqueeze(-1)
            
            assert padded_out_one_hots.requires_grad
            
            # could there be an implementation problem here? 
            embedded_sequences = torch.matmul(padded_out_one_hots, embedding_weights)

            attention_mask = torch.ones(padded_out_one_hots.size(0), padded_out_one_hots.size(1), dtype=torch.bool).to(DEVICE)
            for sequence_num, length in enumerate(sequence_lengths):
                attention_mask[sequence_num, length:] = 0

            stacked_labels = target.repeat(BATCH_SIZE, 1)
            # print('STACKED LABELS')
            # print(stacked_labels)

            judgment_logits = discriminator(inputs_embeds=embedded_sequences, attention_mask=attention_mask, return_dict=False)[0]
            print('JUDGMENT LOGITS')
            print(judgment_logits)

            loss_gen = criterion(judgment_logits, stacked_labels)
            print('LOSS FOR GEN')
            print(loss_gen)

            # pause_for_revision = input('Time for calculation... ')

            loss_gen.backward()
            opt_gen.step()

            # DISCRIMINATOR TRAINING! 

            real_input_ids = batch['input_ids'].to(DEVICE)
            real_attention_mask = batch['attention_mask'].to(DEVICE)
            real_labels = batch['labels'].to(DEVICE)

            opt_disc.zero_grad()
            real_judgments = discriminator(real_input_ids, attention_mask=real_attention_mask, labels=real_labels)
            fake_labels = torch.ones_like(real_labels) * 5
            fake_labels = fake_labels.to(DEVICE)
            fake_judgments = discriminator(inputs_embeds=embedded_sequences.detach(), attention_mask=attention_mask, labels=fake_labels)

            avg_disc_loss = (5 * real_judgments.loss + fake_judgments.loss) / 6  # same weighting as in warm-up stage
            avg_disc_loss.backward()
            opt_disc.step()
            real_losses_disc.append(real_judgments.loss.item())
            fake_losses_disc.append(fake_judgments.loss.item())
            avg_losses_disc.append(avg_disc_loss.item())

            losses_gen.append(loss_gen.item())

            generator.eval()
            
            with torch.no_grad():
                index_sequence = torch.tensor([[tokenizer.cls_token_id]] * 100).to(DEVICE)
                for _ in range(64 - 1):
                    _, one_hots = generator(index_sequence, requires_gumbel_out=True)
                    next_token_id = torch.argmax(one_hots, dim=1).unsqueeze(dim=1)
                    index_sequence = torch.cat([index_sequence, next_token_id], dim=1)
                index_sequence = pad_out_tokens_after_eos(index_sequence, tokenizer.sep_token_id, tokenizer.pad_token_id)
                for ind in range(100):
                    print(tokenizer.decode(index_sequence[ind], skip_special_tokens=True))
            
            generator.train()

            # print(f'G-loss: {loss_gen.item():.2f}\t\tD-loss: {avg_disc_loss.item():.2f}')
        
    
    # save model? 

    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.plot(range(len(losses_gen)), losses_gen)
    plt.plot(range(len(avg_losses_disc)), avg_losses_disc)
    plt.plot(range(len(real_losses_disc)), real_losses_disc)
    plt.plot(range(len(fake_losses_disc)), fake_losses_disc)
    plt.legend(('GEN', 'DISC', 'DISC-real', 'DISC-fake'))
    plt.savefig(f'{INIT_STYLE}{INFL_STYLE}_loss_trends.png')
    plt.clf()


if __name__=='__main__':
    main()
