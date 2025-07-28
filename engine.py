import torch 
import logging 
from tqdm import tqdm 
import numpy as np 
from sklearn.metrics import accuracy_score , f1_score , roc_curve


def calculate_eer(y_true , y_scores):
    fpr , tpr , thresholds = roc_curve(y_true , y_scores , pos_label=1)
    fnr = 1 - tpr 
    abs_diffs = np.abs(fpr - fnr)
    eer_index = np.argmin(abs_diffs)
    eer = (fpr[eer_index] + fnr[eer_index])/2
    
    return eer 


def train_one_epoch(model , dataloader , criterion , optimizer , device , grad_clip_norm):

    model.train()
    total_loss = 0 

    pbar = tqdm(dataloader , desc = "Training")
    for inputs , labels in pbar :
        if inputs is None : continue
        inputs , labels = inputs.to(device) , labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze(-1)

        loss = criterion(outputs , labels)
        if torch.isnan(loss):
            logging.error("NaN loss detected. Stopping training for this epoch")
            return -1 

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters() , max_norm = grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss' : f'{loss.item():.4f}'})

    return total_loss/len(dataloader)


