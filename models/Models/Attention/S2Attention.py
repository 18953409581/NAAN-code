import numpy as np
import torch
from torch import nn
from torch.nn import init



class Channel_Att1(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att1, self).__init__()
        self.channels = channels

        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = torch.sigmoid(x) * residual  #

        return x



# https://arxiv.org/abs/2108.01072
def spatial_shift1(x):
    b,w,h,c = x.size()
    x[:,1:,:,:c//4] = x[:,:w-1,:,:c//4]
    x[:,:w-1,:,c//4:c//2] = x[:,1:,:,c//4:c//2]
    x[:,:,1:,c//2:c*3//4] = x[:,:,:h-1,c//2:c*3//4]
    x[:,:,:h-1,3*c//4:] = x[:,:,1:,3*c//4:]
    return x


def spatial_shift2(x):
    b,w,h,c = x.size()
    x[:,:,1:,:c//4] = x[:,:,:h-1,:c//4]
    x[:,:,:h-1,c//4:c//2] = x[:,:,1:,c//4:c//2]
    x[:,1:,:,c//2:c*3//4] = x[:,:w-1,:,c//2:c*3//4]
    x[:,:w-1,:,3*c//4:] = x[:,1:,:,3*c//4:]
    return x


class SplitAttention(nn.Module):
    def __init__(self,channel=512,k=2):
        super().__init__()
        self.channel=channel
        self.k=k
        self.mlp1=nn.Linear(channel,channel,bias=False)
        self.gelu=nn.GELU()
        self.mlp2=nn.Linear(channel,channel*k,bias=False)
        self.softmax=nn.Softmax(1)
    
    def forward(self,x_all):
        b,k,h,w,c=x_all.shape
        x_all=x_all.reshape(b,k,-1,c) 
        a=torch.sum(torch.sum(x_all,1),1) 
        hat_a=self.mlp2(self.gelu(self.mlp1(a))) 
        hat_a=hat_a.reshape(b,self.k,c) 
        bar_a=self.softmax(hat_a) 
        attention=bar_a.unsqueeze(-2) 
        out=attention*x_all 
        out=torch.sum(out,1).reshape(b,h,w,c)
        return out


class S2Attention(nn.Module):
    # # AIEAGNY
    def __init__(self, channels=512 ):
        super().__init__()
        self.mlp1 = nn.Linear(channels,channels*3)
        self.mlp2 = nn.Linear(channels,channels)
        self.split_attention = SplitAttention()

    def forward(self, x):
        b,c,w,h = x.size()
        x=x.permute(0,2,3,1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:,:,:,:c])
        x2 = spatial_shift2(x[:,:,:,c:c*2])
        x3 = x[:,:,:,c*2:]
        x_all=torch.stack([x1,x2,x3],1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        x=x.permute(0,3,1,2)
        return x


class XCAttention(nn.Module):
    # # AIEAGNY
    def __init__(self, channels = 4 ):
        super().__init__()
        self.mlp1 = nn.Linear(channels,channels*3)
        self.mlp2 = nn.Linear(channels,channels)
        self.split_attention = SplitAttention(channels)
        self.chann =  Channel_Att1(channels)

    def forward(self, x):
        b,c,w,h = x.size()
        identy = x
        x=x.permute(0,2,3,1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:,:,:,:c])
        x2 = spatial_shift2(x[:,:,:,c:c*2])
        x3 = self.chann(identy)
        x_all=torch.stack([x1,x2],1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        x=x.permute(0,3,1,2)
        x = x * x3
        return x



if __name__ == '__main__':
    tt = torch.rand(2,4,64,64)
    win =  XCAttention(4)
    out  =  win(tt)
    print(out.size())