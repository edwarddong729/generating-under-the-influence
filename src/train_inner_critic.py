import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.nn import functional as F
from tqdm import trange
from transformers import BertConfig, BertForSequenceClassification

from dataloaders import inner_critic_training_dataloaders


def main():

    BATCH_SIZE = 128
    MAX_EPOCHS = 60
    TRAIN_SAMPLES_PER_STYLE = 3584  # this means 3584 * 5 total training samples
    VAL_SAMPLES_PER_STYLE = 256  # this means 256 * 5 total validation samples
    EVAL_SAMPLES_PER_STYLE = 256  # this means 256 * 5 total evaluation samples
    LEARNING_RATE = 4e-6
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {DEVICE}')
    torch.manual_seed(15324)  # for reproducibility

    # Load MacBERTh configuration and set dropout regularization hyperparam for classification head
    config = BertConfig.from_pretrained('emanjavacas/MacBERTh', num_labels=5)
    config.classifier_dropout = 0.4

    # Load warm-started model fitted with randomly initiated classification head
    # All layers will be fine-tuned
    model = BertForSequenceClassification.from_pretrained('emanjavacas/MacBERTh', config=config)
    model.to(DEVICE)
    for _, param in model.named_parameters():
        param.requires_grad = True
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Load dataloaders
    train_dl, val_dl, eval_dl = inner_critic_training_dataloaders(TRAIN_SAMPLES_PER_STYLE,
                                                                  VAL_SAMPLES_PER_STYLE,
                                                                  EVAL_SAMPLES_PER_STYLE,
                                                                  BATCH_SIZE)
    print('Length of train dataloader:', str(len(train_dataloader)))
    print('Length of val dataloader:', str(len(val_dataloader)))
    print('Length of eval dataloader:', str(len(eval_dataloader)))

    # Set trackers
    train_losses = []
    val_losses = []
    best_loss = float('inf')

    for epoch in trange(MAX_EPOCHS):
        
        print(f'---------------------- Starting EPOCH {epoch} ----------------------')

        train_loss = 0.0
        model.train()
        for batch_num, batch in enumerate(train_dl):
            
            # Take in batched samples
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            # Compute batch loss and then backpropagate it
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Periodically print out average training loss in epoch so far
            if (batch_num + 1) % 10 == 0:
                average_train_loss = train_loss / (batch_num + 1)
                print(f'Epoch {epoch}, Batch {batch_num+1}/{len(train_dl)}, Average Loss during Epoch: {average_train_loss:.3f}')

        # Keep track of epoch-end average training losses
        train_losses.append(average_train_loss)

        # Begin validation loop, no backprop
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for val_batch_num, val_batch in enumerate(val_dl):
                
                # Take in batched samples
                input_ids = val_batch['input_ids'].to(DEVICE)
                attention_mask = val_batch['attention_mask'].to(DEVICE)
                labels = val_batch['labels'].to(DEVICE)

                # Compute batch loss
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

            # Compute average validation loss and save model if best so far
            average_val_loss = val_loss / len(val_dl)
            print(f'Epoch {epoch}, Validation Loss: {average_val_loss:.3f}')
            if average_val_loss < best_loss:
                best_loss = average_val_loss
                model.save_pretrained('inner_critic')
                print(f'New best model saved at end of epoch!')

            val_losses.append(average_val_loss)

        # Check early stopping criterion (two consecutive increases in validation loss)
        if epoch > 1:
            if val_losses[-1] > val_losses[-2] and val_losses[-2] > val_losses[-3]:
                print('Stopping early because val loss worsened two epochs in a row.')
                break

    # Visualize training and validation loss trajectories on the same plot
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.legend()
    plt.savefig('inner_critic_training.png')
    plt.clf()
    
    # Evaluate best inner critic on held-out data
    model = BertForSequenceClassification.from_pretrained('inner_critic')
    model.to(DEVICE)
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in eval_dl:
            inputs = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            # Get model predictions 
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).cpu().numpy()

            # Save both true labels and predicted labels in respective lists
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions)
    
    # Analyze overall as well as by-label performance 
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels))

    # Print confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)


if __name__=='__main__':
    main()
