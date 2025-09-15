from torch import nn
import torch.nn.functional as F

from models.modules.AdaCos import AdaCos


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.CNN = nn.Sequential(
                nn.Conv1d(1, 80, 100, 5),
                nn.BatchNorm1d(80),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(80, 80, 50, 5),
                nn.BatchNorm1d(80),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(80, 80, 25, 2),
                nn.BatchNorm1d(80),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
        # self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        return self.CNN(x)

class PXRDResNet(nn.Module):
    def __init__(self, block, num_blocks, channels=[64, 128, 256, 512]):
        super(PXRDResNet, self).__init__()
        self.in_channels = channels[0]

        self.stem = nn.Sequential(
            nn.Conv1d(1, self.in_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
            # nn.MaxPool1d(kernel_size=3, stride=2, padding=1) # 初始降维
        )

        self.layer1 = self._make_layer(block, channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, channels[1], num_blocks[1], stride=2) # stride=2 进行降维
        self.layer3 = self._make_layer(block, channels[2], num_blocks[2], stride=2) # stride=2 进行降维
        self.layer4 = self._make_layer(block, channels[3], num_blocks[3], stride=2) # stride=2 进行降维


    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # 主路径
        self.main_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.main_path(x) + self.shortcut(x)
        return F.relu(out)

class Predictor(nn.Module):
    def __init__(self, in_features, out_features):
        super(Predictor, self).__init__()
        self.adacos = nn.Sequential(nn.Flatten(),
                                    nn.Linear(in_features, 2300), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Linear(2300, 1150), nn.ReLU(), nn.Dropout(0.5),
                                    AdaCos(1150,out_features))
        self.MLP = nn.Sequential(nn.Flatten(),
                                 nn.Linear(in_features, 2300), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(2300, 1150), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(1150, out_features))

    def forward(self, x):
        # return self.adacos(x)
        return self.MLP(x)
    

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.net = PXRDResNet(ResidualBlock, [1, 2, 2, 1])
        mlp_in_features = 272384
        if args.task == 'spg':
            self.MLP = Predictor(mlp_in_features, 230)
        elif args.task == 'crysystem':
            self.MLP = Predictor(mlp_in_features, 7)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.interpolate(x,size=8500,mode='linear', align_corners=False)
        x = self.net(x)
        # print(x.shape)
        x = self.MLP(x)
        return x