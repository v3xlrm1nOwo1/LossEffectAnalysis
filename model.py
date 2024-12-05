import torch.nn as nn



class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.c1 = nn.Linear(28* 28, 256)
        self.c2 = nn.Linear(256, 128)
        self.c3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.flatten(x)
        out = self.relu(self.c1(x))
        out = self.relu(self.c2(out))
        out = self.sigmoid(self.c3(out))
        
        return out
    