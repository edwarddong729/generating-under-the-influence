import torch
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from tqdm import trange

from custom_models import CustomGenerator
from dataloaders import gen_warmup_dataloaders


def configure_models():

    tokenizer = BertTokenizer.from_pretrained('emanjavacas/MacBERTh')
    vocab_size = len(tokenizer)
    generator = CustomGenerator(vocab_size)

    # Iterate through the parameters and set requires_grad based on the layer index
    # cuda memory constraint 
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
        else:
            param.requires_grad = True

    return tokenizer, generator
    

def masked_ce_loss(logits, targets, masks):
    
    # Compute the standard cross-entropy loss
    ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')

    # Apply the mask to the loss
    masked_loss = ce_loss * masks

    # Compute the total loss by summing over non-padding tokens
    total_loss = masked_loss.sum()

    # Normalize the loss by the number of non-padding tokens
    num_non_padding_tokens = masks.sum()

    return total_loss, total_loss / num_non_padding_tokens


def main():

    # Set experimental hyperparameters
    TRAIN_SAMPLES_PER_AUTHOR = 3296 # 1536
    VAL_SAMPLES_PER_AUTHOR = 512 # 384
    MAX_EPOCHS = 60
    BATCH_SIZE = 32 # 64 too big for memory
    LR = 2e-5  # consider sending down to 1e-5 because there's a lot more signal now
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {DEVICE}')
    torch.manual_seed(15324)

    # Configure tokenizer and models
    tokenizer, generator = configure_models()
    generator.to(DEVICE)
    print('Model successfully initialized.')

    # Assign optimizers
    opt_gen = torch.optim.AdamW(generator.parameters(), lr=LR)

    # Retrieve dataloaders
    train_dataloader, val_dataloader = gen_warmup_dataloaders('achebe', TRAIN_SAMPLES_PER_AUTHOR, VAL_SAMPLES_PER_AUTHOR, BATCH_SIZE)

    # complication: sometimes indices, sometimes embeddings

    train_avg_losses = []
    val_avg_losses = []
    
    best_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in trange(MAX_EPOCHS):
        print(f'---------------------- Starting EPOCH {epoch} ----------------------')

        epoch_train_loss = 0.0
        generator.train()
        for batch_inputs, batch_labels, loss_masks in train_dataloader:
            batch_inputs = batch_inputs.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            batch_loss_masks = loss_masks.to(DEVICE)

            opt_gen.zero_grad()
            loss_for_backprop = 0.0
            loss_for_monitoring = 0.0

            for i in range(batch_inputs.size(1)):
                
                # teacher forcing
                step_inputs = batch_inputs[:, :i + 1]
                step_labels = batch_labels[:, i]
                step_masks = batch_loss_masks[:, i]

                # Forward pass
                logits, _ = generator(step_inputs)
                unnormalized_loss, normalized_loss = masked_ce_loss(logits, step_labels, step_masks)
                loss_for_backprop += normalized_loss
                loss_for_monitoring += unnormalized_loss

            loss_for_backprop.backward()
            opt_gen.step()

            epoch_train_loss += loss_for_monitoring.item()

        # end of epoch
        epoch_train_loss /= len(train_dataloader)
        train_avg_losses.append(epoch_train_loss)
        print(f'Averaged training loss across epoch {epoch}: {epoch_train_loss:.2f}')

        # SAMPLE GENERATION + VALIDATION LOOP
        epoch_end_val_loss = 0.0
        generator.eval()  # Set the generator model to evaluation mode

        with torch.no_grad():  # Disable gradient calculation during validation and sample generation

            for batch_inputs, batch_labels, loss_masks in val_dataloader:
                batch_inputs = batch_inputs.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                batch_loss_masks = loss_masks.to(DEVICE)

                loss_for_monitoring = 0.0

                for i in range(batch_inputs.size(1)):
                    step_inputs = batch_inputs[:, :i + 1]
                    step_labels = batch_labels[:, i]
                    step_masks = batch_loss_masks[:, i]

                    logits, _ = generator(step_inputs)

                    unnormalized_loss, _ = masked_ce_loss(logits, step_labels, step_masks)
                    
                    loss_for_monitoring += unnormalized_loss

                epoch_end_val_loss += loss_for_monitoring.item()

            # Calculate average validation loss
            epoch_end_val_loss /= len(val_dataloader)

            val_avg_losses.append(epoch_end_val_loss)
            print(f'Averaged validation loss at END of epoch: {epoch_end_val_loss:.2f}')

            if epoch_end_val_loss < best_loss:
                print('New best validation loss! Saving current model...')
                best_loss = epoch_end_val_loss
                # torch.save(generator, 'best_model.pth') # check this
                torch.save(generator.state_dict(), 'A_state_dict.pth')
                # generator.save_pretrained('best_model')
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            # Generate 5 sample sentences
            print('Sixteen sample outputs:')
            for i in range(16):
                sample_sequence = torch.tensor([[tokenizer.cls_token_id]]).to(DEVICE)

                for j in range(64 - 1):
                    
                    _, one_hots = generator(sample_sequence, requires_gumbel_out=True)

                    next_token_id = torch.argmax(one_hots, dim=1).unsqueeze(dim=0)
                    sample_sequence = torch.concat((sample_sequence, next_token_id), dim=1)

                    if next_token_id.item() == tokenizer.sep_token_id:
                        break

                print(tokenizer.decode(sample_sequence.squeeze(), skip_special_tokens=True))
        
        if epochs_since_improvement == 5:
            print('Stopping early because val loss did not improve in 4 epochs.')
            break

    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.plot(range(len(train_avg_losses)), train_avg_losses, label='Training Loss')
    plt.plot(range(len(val_avg_losses)), val_avg_losses, label='Validation Loss')
    plt.legend()
    plt.savefig('A_training.png')
    plt.clf()


if __name__=='__main__':
    main()
