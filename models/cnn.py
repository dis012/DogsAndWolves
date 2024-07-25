import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_chanel=3, num_classes=2):
        super(CNN, self).__init__()
        # Convolution layers (4 conv layers with max pooling after each and batch normalization after each conv layer)
        self.conv1 = nn.Conv2d(in_channels=in_chanel, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) # pooling with kernel size 2 and stride 2 reduces each dimension by factor 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn4 = nn.BatchNorm2d(128)
        
        # Fully connected layers with dropout (probability of 0.5)
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(128*14*14, 1024) # 14x14 is the size of the tensor after 4 max pooling layers and 128 is the number of output channels from the last conv layer
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 2) # Output layer for binary classification with softmax

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 128*14*14) # Flatten the tensor before FC layer
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x