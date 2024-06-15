import torch
import torch.nn as nn

class MixMatchLoss(nn.Module):
    
    def __init__(self, class_num, lambda_u):
        super(MixMatchLoss, self).__init__()
        self.cel = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.class_num = class_num
        self.coeff = lambda_u
        
    def forward(self, outputs, labels, src):
        X_outputs = outputs[src==0]
        X_labels = labels[src==0]
        U_outputs = outputs[src==1]
        U_labels = labels[src==1]
        if len(X_labels > 0):
            X_loss = self.cel(X_outputs, X_labels)
        else:
            X_loss = torch.tensor([0.0]).cuda()
        if len(U_labels > 0):
            U_loss = self.cel(U_outputs, U_labels) / self.class_num
        else:
            U_loss = torch.tensor([0.0]).cuda()
        return X_loss + self.coeff * U_loss