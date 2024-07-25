import numpy as np
import matplotlib.pyplot as plt

def compute_roc(labels, probs, num_thresholds=100):
    thresholds = np.linspace(0, 1, num_thresholds)
    tpr = np.zeros(num_thresholds)
    fpr = np.zeros(num_thresholds)

    # Loop through all thresholds and compute True Positive Rate and False Positive Rate
    for i, threshold in enumerate(thresholds):
        tp = np.sum((probs >= threshold) & (labels == 1))
        fp = np.sum((probs >= threshold) & (labels == 0))
        fn = np.sum((probs < threshold) & (labels == 1))
        tn = np.sum((probs < threshold) & (labels == 0))

        tpr[i] = tp / (tp + fn)
        fpr[i] = fp / (fp + tn)

    return fpr, tpr, thresholds

# Compute the area under the ROC curve to quantify the performance of the model
def compute_auc(fpr, tpr):
    sorted_indices = np.argsort(fpr)
    sorted_fpr = fpr[sorted_indices]
    sorted_tpr = tpr[sorted_indices]
    return np.trapz(sorted_tpr, sorted_fpr)

def plot_roc(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()