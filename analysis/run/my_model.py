import torch
from torch import nn,optim
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
import utils
#import Sift

class Net(nn.Module):
    def __init__(self, in_count, in_dim, n_hidden_1=512, n_hidden_2=512, merge_dim=512):
        super(Net, self).__init__()
        self.__model_name__ = 'Invisible Eye'
        self.in_count = in_count
        for i in range(in_count):
            setattr(self, 'layer_' + str(i), nn.Sequential(
                nn.Linear(np.product(in_dim[:2]), n_hidden_1), nn.ReLU(True),
                nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True)
                ))
        self.merge_regre_layer = nn.Sequential(
            nn.Linear(n_hidden_2 * in_count, merge_dim),
            nn.ReLU(True),
            nn.Linear(merge_dim, 2)
            )

    def input_reset(self, full_reset=True):
        return

    def forward(self, x):
        # TODO: optimize this?
        batch,channel = x.shape[0:2]
        x = x.permute(1, 0, 2, 3)
        x = x.view(channel, batch, -1)
        out = torch.cat([getattr(self, 'layer_' + str(i))(x[i]) for i in range(self.in_count)], 1)
        return self.merge_regre_layer(out)

def get_model(cam_count, cam_dim):
    net = utils.wrap_model(Net(cam_count, cam_dim))
    return net, utils.wrap_optimizer(optim.Adagrad(net.parameters(), lr = 0.005))
