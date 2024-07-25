import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torchvision import transforms
from models.cnn import CNN
from models.early_stopping import EarlyStopping
from models.custom_dataset import DogsAndWolvesDataset
from utils.functions import mean_std, plot_loss
from scripts.evaluate import evaluate_model

def train_model(data_path, batch_size=16, num_epochs=50, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    temp_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

    temp_data = DogsAndWolvesDataset(data_path, temp_transform)

    mean_calc, std_calc = mean_std(temp_data)
    
    # Define transformations with augmentation
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_calc.tolist(), std=std_calc.tolist())
    ])

    dataset = DogsAndWolvesDataset(data_path, transform=transform)

    # Split the dataset into train validate and test
    train_size = int(0.8*len(dataset))
    val_size = int(0.1*len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # Define the model, loss function and optimizer
    model = CNN().to(device)

    # Select criterion and optimizer
    criterion = nn.CrossEntropyLoss() # Cross entropy loss for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # Adam optimizer with L2 regularization

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Early stopping
    early_stopping = EarlyStopping(patience=5, verbose=True)

    training_loss = []
    validation_loss = []
    val_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0

        # loop through all the batches in the training dataset
        for i, (img, label) in enumerate(train_loader):
            # move the data to device
            img = img.to(device)
            label = label.to(device)

            # forward pass
            outputs = model(img)
            loss = criterion(outputs, label)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the training loss
            running_loss += loss.item()

        # Avrage training loss for each epoch
        mean_train_loss = running_loss/len(train_loader)
        training_loss.append(mean_train_loss)
        
        # Validation at the end of each epoch
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            all_val_loss = [] 
            for img, label in val_loader:
                img = img.to(device)
                label = label.to(device)

                outputs = model(img)
                loss = criterion(outputs, label)

                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                all_val_loss.append(loss.item()) 

            # Print the loss and accuracy at the end of each epoch
            mean_val_loss = sum(all_val_loss)/len(all_val_loss)
            validation_loss.append(mean_val_loss)
            accuracy = 100 * correct / total
            val_accuracy.append(accuracy)

            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Validation Loss: {mean_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

            # Early stopping
            early_stopping(mean_val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Step the scheduler
        scheduler.step() 

    # Save the final model weights
    torch.save(model.state_dict(), '/path/to/weights/final_model_weights.pth') # Make sure to change the path!
    print("Model weights saved successfully!")

    plot_loss(training_loss, validation_loss)

    # Evaluate the model on the test set
    evaluate_model(model, test_loader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN to classify dogs and wolves.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    args = parser.parse_args()

    train_model(data_path=args.data_path, batch_size=args.batch_size, num_epochs=args.num_epochs, learning_rate=args.learning_rate)
