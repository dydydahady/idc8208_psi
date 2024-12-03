import torch
import torch.nn as nn
import torch.nn.functional as F

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


class LSTMIntBbox(nn.Module):
    def __init__(self, input_size, hidden_dim, output_dim, observe_length, dropout=0.5, n_layers=1):
        super(LSTMIntBbox, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.observe_length = observe_length
        self.dropout = dropout
        self.n_layers = n_layers

        # Explicitly define self.backbone as None
        self.backbone = None

        # Define return_sequence as an argument
        self.return_sequence = False  # Default value

        # Define a fully connected layer for intent prediction
        self.intent_predictor = nn.Linear(hidden_dim, output_dim)  # Hidden to output (binary classification)

        # Example layers
        self.encoder = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            dropout=dropout if n_layers > 1 else 0,  # Avoid dropout warning
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        # Access the correct key for bounding boxes
        x = data.get('bboxes', None)
        if x is None:
            raise ValueError("Input data dictionary does not contain the key 'bboxes'")

        # Convert the input tensor to float32 if it's not already
        x = x.float()  # This ensures the tensor has dtype torch.float32

        # Pass the bounding boxes (x) through the encoder
        enc_output, _ = self.encoder(x)  # Example LSTM output
        if self.return_sequence:
            enc_last_output = enc_output
        else:
            enc_last_output = enc_output[:, -1, :]  # bs x hidden_dim

        # Pass through the intent predictor (you may need to modify this as per your model)
        intent_pred = self.intent_predictor(enc_last_output)
        return intent_pred


    def build_optimizer(self, args):
        params = list(self.parameters())  # Default to all parameters
        if self.backbone is not None:  # Check if self.backbone exists
            params += list(self.backbone.parameters())
        
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        return optimizer, scheduler

    def lr_scheduler(self, cur_epoch, args, gamma=10, power=0.75):
        decay = (1 + gamma * cur_epoch / args.epochs) ** (-power)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr0'] * decay
            param_group['weight_decay'] = 1e-3
            param_group['momentum'] = 0.9
            param_group['nesterov'] = True
        return

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)

    def predict_intent(self, data):
        bbox = data['bboxes'][:, :self.args.observe_length, :].type(FloatTensor)
        # global_imgs = data['images']
        # local_imgs = data['cropped_images']
        dec_input_emb = None # as the additional emb for intent predictor
        # bbox: shape = [bs x observe_length x enc_input_dim]
        assert bbox.shape[1] == self.observe_length

        # 1. backbone feature (to be implemented)
        if self.backbone is not None:
            pass

        # 2. intent prediction
        intent_pred = self.intent_predictor(bbox, dec_input_emb)
        # bs x int_pred_len=1 x int_dim=1
        return intent_pred.squeeze()

class LSTMInt(nn.Module):
    def __init__(self, args, model_opts):
        super(LSTMInt, self).__init__()

        enc_in_dim = model_opts['enc_in_dim']
        enc_out_dim = model_opts['enc_out_dim']
        # dec_in_emb_dim = model_opts['dec_in_emb_dim']
        # dec_out_dim = model_opts['dec_out_dim']
        output_dim = model_opts['output_dim']
        n_layers = model_opts['n_layers']
        dropout = model_opts['dropout']

        self.args = args

        self.enc_in_dim = enc_in_dim  # input bbox+convlstm_output context vector
        self.enc_out_dim = enc_out_dim
        self.encoder = nn.LSTM(
            input_size=self.enc_in_dim,
            hidden_size=self.enc_out_dim,
            num_layers=n_layers,
            batch_first=True,
            bias=True
        )

        self.output_dim = output_dim  # 2/3: intention; 62 for reason; 1 for trust score; 4 for trajectory.

        self.fc = nn.Sequential(
            nn.Linear(self.enc_out_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, self.output_dim)

        )

        if model_opts['output_activation'] == 'tanh':
            self.activation = nn.Tanh()
        elif model_opts['output_activation'] == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

        self.module_list = [self.encoder, self.fc] #, self.fc_emb, self.decoder
        # self._reset_parameters()
        # assert self.enc_out_dim == self.dec_out_dim

    def forward(self, enc_input, dec_input_emb=None):
        enc_output, (enc_hc, enc_nc) = self.encoder(enc_input)
        # because 'batch_first=True'
        # enc_output: bs x ts x (1*hiden_dim)*enc_hidden_dim --- only take the last output, concatenated with dec_input_emb, as input to decoder
        # enc_hc:  (n_layer*n_directions) x bs x enc_hidden_dim
        # enc_nc:  (n_layer*n_directions) x bs x enc_hidden_dim
        enc_last_output = enc_output[:, -1:, :]  # bs x 1 x hidden_dim
        output = self.fc(enc_last_output)
        outputs = output.unsqueeze(1) # bs x 1 --> bs x 1 x 1
        return outputs  # shape: bs x predict_length x output_dim, no activation


    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
