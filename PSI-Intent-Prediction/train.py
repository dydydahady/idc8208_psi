import collections

from test import validate_intent
import torch
import numpy as np
from torch import FloatTensor
import os
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def train_intent(model, optimizer, scheduler, train_loader, val_loader, args, recorder, writer, criterions):    # pos_weight = torch.tensor(args.intent_positive_weight).to(device) # n_neg_class_samples(5118)/n_pos_class_samples(11285)
    pos_weight = torch.tensor(args.intent_positive_weight).to(device)  # Weight for the positive class
    criterions = {
        'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight).to(device),
        'MSELoss': torch.nn.MSELoss(reduction='none').to(device),
        'BCELoss': torch.nn.BCELoss().to(device),
        'CELoss': torch.nn.CrossEntropyLoss(),
    }
    epoch_loss = {'loss_intent': [], 'loss_traj': []}

    for epoch in range(1, args.epochs + 1):
        niters = len(train_loader)
        recorder.train_epoch_reset(epoch, niters)
        epoch_loss = train_intent_epoch(epoch, model, optimizer, criterions, epoch_loss, train_loader, args, recorder, writer)
        scheduler.step()

        if epoch % 1 == 0:
            print(f"Train epoch {epoch}/{args.epochs} | epoch loss: "
                  f"loss_intent = {np.mean(epoch_loss['loss_intent']): .4f}")
            
            # Log the training loss to TensorBoard
            writer.add_scalar('Loss/train_loss', np.mean(epoch_loss['intent_loss']), epoch)

        if (epoch + 1) % args.val_freq == 0:
            print(f"Validate at epoch {epoch}")
            niters = len(val_loader)
            recorder.eval_epoch_reset(epoch, niters)
            validate_intent(epoch, model, val_loader, args, recorder, writer)

            # Log learning rate to TensorBoard
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)


        torch.save(model.state_dict(), args.checkpoint_path + f'/latest.pth')


def train_intent_epoch(epoch, model, optimizer, criterions, epoch_loss, dataloader, args, recorder, writer):
    model.train()
    batch_losses = collections.defaultdict(list)

    # Initialize epoch_loss keys to avoid KeyError
    if 'intent_loss' not in epoch_loss:
        epoch_loss['intent_loss'] = []
    if 'fine_tune_loss' not in epoch_loss and args.fine_tune:
        epoch_loss['fine_tune_loss'] = []
    if 'total_loss' not in epoch_loss:
        epoch_loss['total_loss'] = []

    for itern, data in enumerate(dataloader):
        # print(f"Data keys: {data.keys()}")
        print(f"Processing Batch {itern}")
        optimizer.zero_grad()

        # Forward pass
        if args.fine_tune:
            # If fine-tuning, the model returns both intent logits and auxiliary features
            intent_logits, aux_features = model(data)
            intent_logits = intent_logits.squeeze(-1)

            # Compute auxiliary loss if fine-tuning
            gt_aux = data['aux_ground_truth']  # Auxiliary ground truth; ensure it exists in your dataset
            fine_tune_loss = criterions['MSELoss'](aux_features, gt_aux)

            # Reduce fine_tune_loss to scalar
            fine_tune_loss = fine_tune_loss.mean()
        else:
            # If not fine-tuning, the model returns only the intent logits
            intent_logits = model(data)
            intent_logits = intent_logits.squeeze(-1)
            fine_tune_loss = 0

        # Compute primary intent loss
        gt_intent = data['intention_binary'][:, args.observe_length].type(torch.FloatTensor).to(device)

        class_weights = args.class_weights
        bce_loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights[1])

        # intent_loss = criterions['BCEWithLogitsLoss'](intent_logits, gt_intent)
        try:
            intent_loss = bce_loss_function(intent_logits, gt_intent)
            batch_losses['intent_loss'].append(intent_loss.item())
        except RuntimeError as e:
            print(f"Error calculating intent loss at Batch {itern}: {str(e)}")
            continue

         # Backward pass
        total_loss = intent_loss
        total_loss.backward()
        optimizer.step()

        # Logging
        batch_losses['total_loss'].append(total_loss.item())
        if itern % args.print_freq == 0:
            print(f"Epoch {epoch}, Batch {itern}: Intent Loss = {intent_loss.item():.4f}, Total Loss = {total_loss.item():.4f}")

            # Log the batch loss to TensorBoard
            writer.add_scalar('Batch Loss/intent_loss', intent_loss.item(), epoch * len(dataloader) + itern)
            writer.add_scalar('Batch Loss/total_loss', total_loss.item(), epoch * len(dataloader) + itern)

    # Record epoch losses
    if 'intent_loss' in batch_losses:
        epoch_loss['intent_loss'].append(np.mean(batch_losses['intent_loss']))
    else:
        print("Warning: 'intent_loss' key not found in batch_losses. Skipping intent_loss logging for this epoch.")
        epoch_loss['intent_loss'].append(0.0)

    epoch_loss['total_loss'].append(np.mean(batch_losses['total_loss']))

    # Log the epoch loss to TensorBoard
    writer.add_scalar('Epoch Loss/intent_loss', np.mean(epoch_loss['intent_loss']), epoch)
    writer.add_scalar('Epoch Loss/total_loss', np.mean(epoch_loss['total_loss']), epoch)

    return epoch_loss