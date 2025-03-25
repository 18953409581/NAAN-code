import torch
import torch.nn as nn

def channel_shuffle(self, x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]



if __name__ == '__main__':
    x = torch.randn(1,4,4,4)
    out1,out2 = channel_shuffle(x)
    print(out1.shape)
    print(out2.shape)