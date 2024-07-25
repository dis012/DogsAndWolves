import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torchvision import transforms
from models.cnn import CNN
from models.custom_dataset import DogsAndWolvesDataset
from utils.functions import mean_std
from utils.visualize import compute_roc, compute_auc, plot_roc

def evaluate_model(model, test_loader, device):
    # Set the model to evaluation mode
    model.eval()

    # Store all the labels and probabilities
    all_labels = []
    all_probs = []

    with torch.no_grad(): # No need to track the gradients
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device)

            outputs = model(img)
            # Apply softmax to output to get predicted probabilities
            probs = F.softmax(outputs, dim=1)[:, 1] # Probability of being a wolf
            
            # Store the labels and probabilities
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    fpr, tpr, thresholds = compute_roc(all_labels, all_probs)

    roc_auc = compute_auc(fpr, tpr)
    
    plot_roc(fpr, tpr, roc_auc)