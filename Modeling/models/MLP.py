import torch
import torch.nn as nn
import torch.optim as optim


class MLPNET(nn.Module):
    def __init__(self, fs, sizes, activation = nn.ReLU()):
        nn.Module.__init__(self)

        self.fs = fs
        self.sizes = sizes
        self.sig = activation
        self.layer = nn.ModuleList()

        gain = 5/3 if isinstance(activation, nn.Tanh) else 1

        for i in range(len(sizes)-1):
            linear = nn.Linear(in_features=sizes[i], out_features=sizes[i+1])

            nn.init.xavier_normal_(linear.weight, gain=gain)
            nn.init.zeros_(linear.bias)

            self.layer.append(linear)

    def forward(self, x):
        for i, layer in enumerate(self.layer):
            x = layer(x)
            if i < len(self.layer) - 1:
                x = self.sig(x)
        
        return x.squeeze()
    
    def train_net(self, X_train, y_train, **kwargs):

        epochs = kwargs['epochs']
        lr = kwargs['lr']
        verbose = kwargs['verbose']

        try:
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
        except:
            X_train = torch.tensor(X_train.values, dtype=torch.float32)
            y_train = torch.tensor(y_train.values, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.train()

        for epoch in range(epochs):
            optimizer.zero_grad()

            preds = self(X_train)
            loss = criterion(preds, y_train)

            loss.backward()
            optimizer.step()

            if epoch % verbose == 0:
                print(f"{self.__class__.__name__} | epoch {epoch} | loss: {loss.item():.4f}")


    def predict(self, X):
        self.eval()
        with torch.no_grad():
            try:
                X_tensor = torch.tensor(X, dtype=torch.float32)
            except:
                X_tensor = torch.tensor(X.values, dtype = torch.float32)    
            preds = self(X_tensor)            
            return preds.numpy()
        
