import numpy as np
import torch
import os
from .intent_modules.model_lstm_int_bbox import LSTMIntBbox

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def build_model(args):
    """
    Build and return the model, optimizer, and scheduler based on the model name in args.
    """
    # Intent models
    if args.model_name == 'lstm_int_bbox':
        model = get_lstm_intent_bbox(args).to(device)
        optimizer, scheduler = model.build_optimizer(args)
        return model, optimizer, scheduler

# 1. Intent prediction
# 1.1 input bboxes only
def get_lstm_intent_bbox(args):
    model_configs = {}
    model_configs['intent_model_opts'] = {
        'enc_in_dim': 4,  # input bbox (normalized OR not) + img_context_feat_dim
        'enc_out_dim': 64,
        'dec_in_emb_dim': None,  # encoder output + bbox
        'dec_out_dim': 64,
        'output_dim': 1,  # intent prediction, output logits, add activation later
        'n_layers': 1,
        'dropout': 0.5,
        'observe_length': args.observe_length,  # 15
        'predict_length': 1,  # only predict one intent
        'return_sequence': False,  # False for reason/intent/trust. True for trajectory
        'output_activation': 'None'  # [None | tanh | sigmoid | softmax]
    }
    args.model_configs = model_configs

    # Extract parameters for clarity
    input_size = model_configs['intent_model_opts']['enc_in_dim']
    hidden_dim = model_configs['intent_model_opts']['enc_out_dim']
    output_dim = model_configs['intent_model_opts']['output_dim']
    observe_length = model_configs['intent_model_opts']['observe_length']
    dropout = model_configs['intent_model_opts']['dropout']
    n_layers = model_configs['intent_model_opts']['n_layers']

    # Instantiate the model
    model = LSTMIntBbox(
        input_size=input_size,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        observe_length=observe_length,
        dropout=dropout,
        n_layers=n_layers
    )
    return model
