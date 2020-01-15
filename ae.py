import torch.nn as nn


class Autoencoder3D(nn.Module):
    def __init__(self):
        super(Autoencoder3D, self).__init__()

        # Create memoryless-blocks for reuse
        self.relu = nn.ReLU(True)
        self.tanh = nn.Softsign()

        ########
        #
        #  Encoder layers:
        #
        ########

        # Layer input image size: 3x16x112x112 ; Kernel size: 1
        # N channels in: 3 ; N channels out: 32 ; Stride: 1
        # Layer output image size: 32x16x112x112 ; Pad: 0
        self.conv1 = nn.Conv3d(3, 32, kernel_size=1, padding=0)

        # Layer input image size: 32x16x112x112 ; Kernel size: 2
        # N channels in: 32 ; N channels out: 32 ; Stride: 1
        # Layer output image size: 32x8x56x56 ; Pad: 0
        self.pool1 = nn.MaxPool3d(2, return_indices=True)

        # Layer input image size: 32x8x56x56 ; Kernel size: 5
        # N channels in: 32 ; N channels out: 32 ; Stride: 1
        # Layer output image size: 32x8x56x56 ; Pad: 2
        self.conv2 = nn.Conv3d(32, 32, kernel_size=5, padding=2)

        # Layer input image size: 32x8x56x56 ; Kernel size: 2
        # N channels in: 32 ; N channels out: 32 ; Stride: 1
        # Layer output image size: 32x4x28x28 ; Pad: 0
        self.pool2 = nn.MaxPool3d(2, return_indices=True)

        # Layer input image size: 32x4x28x28 ; Kernel size: 5
        # N channels in: 32 ; N channels out: 32 ;  Stride: 1
        # Layer output image size: 32x4x28x28 ; Pad: 2
        self.conv3 = nn.Conv3d(32, 64, kernel_size=5, padding=2)

        # Layer input image size: 32x4x28x28 ; Kernel size: 2
        # N channels_in: 32 ; N channels_out: 32 ;  Stride: 1
        # Layer output image size: 32x2x14x14
        self.pool3 = nn.MaxPool3d(2, return_indices=True)

        # Layer input image size: 32x2x14x14 ; Kernel size: 1
        # N channels in: 32 ; N channels out: 1 ;  Stride: 1
        # Layer output image size: 32x2x14x14 ; Pad: 0
        self.conv4 = nn.Conv3d(64, 64, kernel_size=1, padding=0)

        ########
        #
        #  Decoder layers:
        #
        ########

        # Layer input image size: 32x2x14x14 ; Kernel size: 3
        # N channels in: 1 ; N channels out: 1 ;  Stride: 1
        # Layer output image size: 32x2x14x14 ; Pad: 1
        self.deconv4 = nn.ConvTranspose3d(64, 64, kernel_size=3, stride=1,
                                          padding=1)

        # Layer input image size: 32x2x14x14 ; Kernel size: 4
        # N channels in: 1 ; N channels out: 32 ;  Stride: 2
        # Layer output image size: 32x4x28x28 ; Pad: 1
        self.deconv3 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2,
                                          padding=1)

        # Layer input image size: 32x4x28x28 ; Kernel size: 4
        # N channels in: 32 ; N channels out: 32 ;  Stride: 2
        # Layer output image size: 32x8x56x56 ; Pad: 1
        self.deconv2 = nn.ConvTranspose3d(32, 32, kernel_size=4, stride=2,
                                          padding=1)

        # Layer input image size: 32x8x56x56 ; Kernel size: 4
        # N channels in: 32 ; N channels out: 32 ;  Stride: 2
        # Layer output image size: 32x16x112x112 ; Pad: 1
        self.deconv1 = nn.ConvTranspose3d(32, 32, kernel_size=4, stride=2,
                                          padding=1)

        # Layer input image size: 32x16x112x112 ; Kernel size: 1
        # N channels in: 32 ; N channels out: 3 ;  Stride: 1
        # Layer output image size: 3x16x112x112 ; Pad: 0
        self.deconv0 = nn.ConvTranspose3d(32, 3, kernel_size=1, stride=1,
                                          padding=0)

    ########
    #
    #  Constructed Encoder
    #
    ########
    def encoder(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x, ind1 = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x, ind2 = self.pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x, ind3 = self.pool3(x)
        x = self.conv4(x)
        x = self.relu(x)
        return x, [ind1, ind2, ind3]

    ########
    #
    #  Constructed Decoder
    #
    ########
    def decoder(self, x, indices):
        ind1, ind2, ind3 = indices
        x = self.deconv4(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.relu(x)
        x = self.deconv1(x)
        x = self.deconv0(x)
        x = self.tanh(x)
        return x

    ########
    #
    #  Forward Propagation
    #
    ########
    def forward(self, x):
        x, indices = self.encoder(x)
        x = self.decoder(x, indices)
        return x