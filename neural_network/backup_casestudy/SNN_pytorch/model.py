import torch.nn as nn
import torch.nn.functional as F
import torch
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            # nn.ReflectionPad1d(1),
            nn.Conv1d(1, 20, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(20),
            
            # nn.ReflectionPad1d(1),
            nn.Conv1d(20, 20, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(20),


            # nn.ReflectionPad1d(1),
            nn.Conv1d(20, 20, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(20),


        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 2))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

