import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from Network import *





def RMSE_PC(model_out,test_output):
    loss_func = nn.MSELoss()
    mse=loss_func(model_out,test_output)
    RMSE_loss=torch.sqrt(mse)
    return RMSE_loss
"""    x = model_out
    y = test_output

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

    print('Pearson Correlation:',cost)
"""

def greedy_decode(model, src, src_mask, max_len):
    memory = model.encode(src, src_mask)
    start=src[:,-1,:]
    ys = start.unsqueeze(2)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        out=out[:,-1,:].unsqueeze(2)
        ys = torch.cat([ys,out], dim=1)
    out = model.decode(memory, src_mask,
                       Variable(ys),
                       Variable(subsequent_mask(ys.size(1))
                                .type_as(src.data)))

    return out

#RMSE_PC(total_out[:,-1],test_output)
#print(total_out[:,-1])
#torch.save(total_out[:,-1], 'model_output1.pt')

