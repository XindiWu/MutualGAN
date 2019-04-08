import torch.nn as nn

'''
modified to fit dataset size
'''


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            
            # input shape is 224 x 224 x 3
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # shape is 55 x 55 x 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # shape is 27 x 27 x 64

            nn.Conv2d(64, 192, kernel_size=5, padding=2), # shape is 27 x 27 x 192
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # shape is 13 x 13 x 192

            nn.Conv2d(192, 384, kernel_size=3, padding=1), # shape is 13 x 13 x 384
            nn.ReLU(),

            nn.Conv2d(384, 256, kernel_size=3, padding=1), # shape is 13 x 13 x 256
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1), # shape is 13 x 13 x 256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2) # shape is 6 x 6 x 256
        )
        '''
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            '''
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 2 * 2, 4096),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes),
        # )

        self.feature = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # 256 * 2 * 2
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),   # 512
            # nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.feature(x)
        return x



class AlexClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(4096, num_classes),  # 512
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class Alex_complete(nn.Module):
    def __init__(self, num_classes=10):
        super(Alex_complete, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
             nn.Dropout(),
             nn.Linear(256 * 2 * 2, 512),
             nn.ReLU(),
             nn.Dropout(),
             nn.Linear(512, 512),
             nn.ReLU(inplace=True),
             nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
