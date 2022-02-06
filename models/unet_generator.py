import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    """
    This is UNet from semantic segmentation homework.
    """
    def __init__(self):
        super().__init__()
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        self.pool0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        self.pool1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        self.pool2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        self.pool3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=1)

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2),
                                            padding=(1, 1), output_padding=(1, 1))
        self.dec_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2),
                                            padding=(1, 1), output_padding=(1, 1))
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=128 * 2, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2),
                                            padding=(1, 1), output_padding=(1, 1))
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64 * 2, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.upsample3 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2),
                                            padding=(1, 1), output_padding=(1, 1))
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32 * 2, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

        self.weight_init(mean=0.0, std=0.02)

    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    def forward(self, x):
        # encoder
        e0 = self.enc_conv0(x)
        e0_up = self.pool0(e0)

        e1 = self.enc_conv1(e0_up)
        e1_up = self.pool1(e1)

        e2 = self.enc_conv2(e1_up)
        e2_up = self.pool2(e2)

        e3 = self.enc_conv3(e2_up)
        e3_up = self.pool3(e3)

        # bottleneck
        b = self.bottleneck_conv(e3_up)

        # decoder
        d0 = self.upsample0(b)
        d0 = self.dec_conv0(torch.cat([d0, e3], dim=1))

        d1 = self.upsample1(d0)
        d1 = self.dec_conv1(torch.cat([d1, e2], dim=1))

        d2 = self.upsample2(d1)
        d2 = self.dec_conv2(torch.cat([d2, e1], dim=1))

        d3 = self.upsample3(d2)
        d3 = self.dec_conv3(torch.cat([d3, e0], dim=1))
        return d3
