# models/build_model.py

import torch
from .intent_modules.model_lstm_int_bbox import TransformerIntBbox

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def build_model(args):
    # Intent models
    if args.model_name == 'transformer_int_bbox':
        model = TransformerIntBbox(
            input_dim=args.input_dim, 
            hidden_dim=args.hidden_dim, 
            output_dim=1,  # Binary classification
            num_layers=args.num_layers,
            nhead=args.nhead,
            dropout=args.dropout,
            observe_length=args.observe_length  # Pass observe_length to the model
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        return model, optimizer, scheduler
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")


