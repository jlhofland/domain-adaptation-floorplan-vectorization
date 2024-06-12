import torch
import torch.nn as nn
import torch.nn.functional as F
from models import human_pose_estimation
from models.residual import Residual

class CubiCasaMMD(nn.Module):
    def __init__(self, classes, latent_dim=512):
        super(CubiCasaMMD, self).__init__()
        self.conv1_ = nn.Conv2d(3, 64, bias=True, kernel_size=7, stride=2, padding=3) # Output: (B, 64, H/2, W/2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.r01 = Residual(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) # Output: (B, 128, H/4, W/4)
        self.r02 = Residual(128, 128)
        self.r03 = Residual(128, 128)
        self.r04 = Residual(128, 256)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: (B, 256, H/8, W/8)
        self.r11_a = Residual(256, 256)
        self.r12_a = Residual(256, 256)
        self.r13_a = Residual(256, 256)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: (B, 256, H/16, W/16)
        self.r21_a = Residual(256, 256)
        self.r22_a = Residual(256, 256)
        self.r23_a = Residual(256, 256)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: (B, 256, H/32, W/32)
        self.r31_a = Residual(256, 256)
        self.r32_a = Residual(256, 256)
        self.r33_a = Residual(256, 256)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: (B, 256, H/64, W/64)
        self.r41_a = Residual(256, 256)
        self.r42_a = Residual(256, 256)
        self.r43_a = Residual(256, 256)

        ###     OLD ARCHITECTURE      ###
        # self.r44_a = Residual(256, 512)
        # self.r45_a = Residual(512, 512)

        ###    NEW ARCHITECTURE      ###
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: (B, 256, H/128, W/128)
        self.r51_a = Residual(256, 256)
        self.r52_a = Residual(256, 256)
        self.r53_a = Residual(256, 256)

        self.maxpool6 = nn.AdaptiveMaxPool2d((1, 1)) # Output: (B, 256, H/256, W/256)
        self.r61_a = Residual(256, 256)
        self.r62_a = Residual(256, 256)
        self.r63_a = Residual(256, 256)
        self.r64_a = Residual(256, 512)
        self.r65_a = Residual(512, 512)

        self.upsample6 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2) # Output: (B, 512, H/128, W/128)
        self.r61_b = Residual(256, 256)
        self.r62_b = Residual(256, 256)
        self.r63_b = Residual(256, 512)
        self.r6_   = Residual(512, 512)

        self.upsample5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2) # Output: (B, 512, H/64, W/64)
        self.r51_b = Residual(256, 256)
        self.r52_b = Residual(256, 256)
        self.r53_b = Residual(256, 512)
        self.r5_   = Residual(512, 512)
        ###   NEW ARCHITECTURE      ###

        self.upsample4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2) # Output: (B, 512, H/32, W/32)
        self.r41_b = Residual(256, 256)
        self.r42_b = Residual(256, 256)
        self.r43_b = Residual(256, 512)

        self.r4_ = Residual(512, 512)
        self.upsample3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2) # Output: (B, 512, H/16, W/16)

        self.r31_b = Residual(256, 256)
        self.r32_b = Residual(256, 256)
        self.r33_b = Residual(256, 512)

        self.r3_ = Residual(512, 512)
        self.upsample2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2) # Output: (B, 512, H/8, W/8)

        self.r21_b = Residual(256, 256)
        self.r22_b = Residual(256, 256)
        self.r23_b = Residual(256, 512)

        self.r2_ = Residual(512, 512)
        self.upsample1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2) # Output: (B, 512, H/4, W/4)

        self.r11_b = Residual(256, 256)
        self.r12_b = Residual(256, 256)
        self.r13_b = Residual(256, 512)

        self.conv2_ = nn.Conv2d(512, 512, bias=True, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3_ = nn.Conv2d(512, 256, bias=True, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4_ = nn.Conv2d(256, classes, bias=True, kernel_size=1)
        self.upsample = nn.ConvTranspose2d(classes, classes, kernel_size=4, stride=4)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_latent=False, return_output=True):
        out = self.conv1_(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool(out)
        out = self.r01(out)
        out = self.r02(out)
        out = self.r03(out)
        out = self.r04(out)

        out1a = self.maxpool1(out)
        out1a = self.r11_a(out1a)
        out1a = self.r12_a(out1a)
        out1a = self.r13_a(out1a)

        out1b = self.r11_b(out)
        out1b = self.r12_b(out1b)
        out1b = self.r13_b(out1b)

        out2a = self.maxpool2(out1a)
        out2a = self.r21_a(out2a)
        out2a = self.r22_a(out2a)
        out2a = self.r23_a(out2a) # Output: (B, 256, H/16, W/16)

        out2b = self.r21_b(out1a)
        out2b = self.r22_b(out2b)
        out2b = self.r23_b(out2b) # Output: (B, 256, H/16, W/16)

        out3a = self.maxpool3(out2a)
        out3a = self.r31_a(out3a)
        out3a = self.r32_a(out3a)
        out3a = self.r33_a(out3a) # Output: (B, 256, H/32, W/32)

        out3b = self.r31_b(out2a)
        out3b = self.r32_b(out3b)
        out3b = self.r33_b(out3b) # Output: (B, 256, H/32, W/32)

        out4a = self.maxpool4(out3a)
        out4a = self.r41_a(out4a)
        out4a = self.r42_a(out4a)
        out4a = self.r43_a(out4a)

        out4b = self.r41_b(out3a)
        out4b = self.r42_b(out4b)
        out4b = self.r43_b(out4b) 

        ### OLD ARCHITECTURE ###
        # out4a = self.r44_a(out4a)
        # out4a = self.r45_a(out4a)

        ### NEW ARCHITECTURE ###
        out5a = self.maxpool5(out4a)
        out5a = self.r51_a(out5a)
        out5a = self.r52_a(out5a)
        out5a = self.r53_a(out5a)

        out5b = self.r51_b(out4a)
        out5b = self.r52_b(out5b)
        out5b = self.r53_b(out5b) 

        out6a = self.maxpool6(out5a)
        out6a = self.r61_a(out6a)
        out6a = self.r62_a(out6a)
        out6a = self.r63_a(out6a)

        # If we do not want the prediction, return the latent space
        if not return_output:
            return None, out6a
        
        # Else, save for later
        else:
            latent = out6a

        out6a = self.r64_a(out6a)
        out6a = self.r65_a(out6a)

        out6b = self.r61_b(out5a)
        out6b = self.r62_b(out6b)
        out6b = self.r63_b(out6b)

        out6_ = self.upsample6(out6a)
        out6 = self._upsample_add(out6_, out6b)
        out6 = self.r6_(out6)

        out5_ = self.upsample5(out6)
        out5 = self._upsample_add(out5_, out5b)
        out5 = self.r5_(out5)
        ### NEW ARCHITECTURE ###

        out4_ = self.upsample4(out5)
        out4 = self._upsample_add(out4_, out4b)
        out4 = self.r4_(out4)

        out3_ = self.upsample3(out4)
        out3 = self._upsample_add(out3_, out3b)
        out3 = self.r3_(out3)

        out2_ = self.upsample2(out3)
        out2 = self._upsample_add(out2_, out2b)
        out2 = self.r2_(out2)

        out1_ = self.upsample1(out2)
        out = self._upsample_add(out1_, out1b)

        out = self.conv2_(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3_(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv4_(out)
        out = self.upsample(out)

        # heatmap channels go trough sigmoid
        out[:, :21] = self.sigmoid(out[:, :21])

        if return_latent:
            return out, latent
        else:
            return out, None

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        if y.shape != x.shape:
            return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y
        else:
            return x + y

    def init_weights(self):
        # Pre-trained network weights from Human pose estimation via Convolutional Part Heatmap Regression
        # https://www.adrianbulat.com/human-pose-estimation MPII
        model = human_pose_estimation.model_1427
        model.load_state_dict(torch.load('./weights/human_pose_estimation.pth'))

        for (src, dst) in zip(model.parameters(), self.parameters()):
            dst[:].data.copy_(src[:].data)
