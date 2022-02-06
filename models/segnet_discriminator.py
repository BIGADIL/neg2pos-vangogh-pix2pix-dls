import torch.nn as nn


class SegNetDiscriminator(nn.Module):
    """
    This is encoder part of SegNet from GAN homework.
    """
    def __init__(self):
        super().__init__()
        self.conv_0 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )

        self.out = nn.Sequential(
            nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(2, 2), padding=0, bias=False)
        )

        self.weight_init(mean=0.0, std=0.02)

    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    def forward(self, x):
        # encoder
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.out(x)
        return x
