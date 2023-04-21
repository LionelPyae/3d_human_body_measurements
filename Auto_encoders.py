import torch
import torch.nn as nn
import torch.optim as optim

# define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(4, 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# load the binary images
binary_images = ... # shape: (num_images, 1, height, width)

# convert the images to a PyTorch tensor
binary_images_tensor = torch.from_numpy(binary_images).float()

# create an instance of the autoencoder
autoencoder = Autoencoder()

# define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# train the autoencoder
num_epochs = 10
for epoch in range(num_epochs):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward pass
    outputs = autoencoder(binary_images_tensor)
    loss = criterion(outputs, binary_images_tensor)

    # backward pass and optimize
    loss.backward()
    optimizer.step()

    # print the loss for this epoch
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# extract the low-dimensional features from the images
encoded_images = autoencoder.encoder(binary_images_tensor).detach().numpy()
