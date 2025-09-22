# model.py
import torch.nn as nn
import torch.nn.functional as F

class PneumoniaDiagnosis(nn.Module):
  def __init__(self):
    super(PneumoniaDiagnosis, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.bn3 = nn.BatchNorm2d(128)
    self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
    self.bn4 = nn.BatchNorm2d(256)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    self.fc1_input_features = 256 * 8 * 8
    self.fc1 = nn.Linear(self.fc1_input_features, 512)
    self.dropout = nn.Dropout(0.5)
    self.fc2 = nn.Linear(512, 2)

  def forward(self, x):
    x = self.pool(F.relu(self.bn1(self.conv1(x))))
    x = self.pool(F.relu(self.bn2(self.conv2(x))))
    x = self.pool(F.relu(self.bn3(self.conv3(x))))
    x = self.pool(F.relu(self.bn4(self.conv4(x))))
    x = x.view(-1, self.fc1_input_features)
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x