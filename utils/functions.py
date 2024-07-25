import torch
import matplotlib.pyplot as plt

# Calculate the mean and std of images for data normalization
def mean_std(dataset):
    mean = torch.zeros(3)
    std = torch.zeros(3)

    for img, _ in dataset:
        mean += img.mean([1,2]) 
        std += img.mean([1,2])
    mean /= len(dataset)
    std /= len(dataset)

    return mean, std

def plot_loss(training_loss, validation_loss):
    # Plot the training and validation loss
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(training_loss, label="train")
    plt.plot(validation_loss, label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()