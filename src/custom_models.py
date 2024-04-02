import torch
from transformers import BertModel


class CustomGenerator(torch.nn.Module):
    """ 
    Generator class that stacks token prediction head on top of pre-trained MacBERTh model
    
    Input dims: batch size * sequence length
    Output dims: batch size * vocab size
    """
    
    def __init__(self, vocab_size):
        super().__init__()
        # parameters will be overriden later on if load_state_dict called
        self.macberth = BertModel.from_pretrained('emanjavacas/MacBERTh')
        self.model_size = self.macberth.config.hidden_size
        self.prediction_head = TokenPredictionHead(self.model_size, vocab_size)

    def forward(self, input_ids, gumbel_tau=2, requires_gumbel_out=False):
        macberth_out = self.macberth(input_ids=input_ids)
        last_token_representation = macberth_out.last_hidden_state[:, -1, :]
        max_pooling, _ = torch.max(macberth_out.last_hidden_state, dim=1)
        pooled_output = macberth_out.pooler_output

        prediction_input = torch.cat((last_token_representation, max_pooling, pooled_output), dim=1)
        return self.prediction_head(prediction_input, gumbel_tau, requires_gumbel_out)


class TokenPredictionHead(torch.nn.Module):
    """
    Custom token prediction head that takes a composite representation of preceding sequence 
    from MacBERTh and passes it through two FF layers 

    Input dims: batch size * three times model size 
    Output dims: batch size * vocab size
    """
    def __init__(self, model_size, vocab_size):
        super().__init__()
        # dropout is deactivated during inference with model.eval()
        self.dropout = torch.nn.Dropout(0.4)
        self.fc1 = torch.nn.Linear(model_size * 3, 324)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.fc2 = torch.nn.Linear(324, vocab_size)

    # gumbel softmax hyperparameters are irrelevant at training stage of generator
    def forward(self, x, tau, gumbel_out):
        x = self.dropout(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.fc2(x)

        if gumbel_out:
            one_hot_x = torch.nn.functional.gumbel_softmax(x, tau=tau, hard=True)
        else:
            one_hot_x = None
        # one_hot_x consists of one-hot outputs (dtype float)
        return x, one_hot_x
