import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from Network import *
import math, copy, time

from test_validation import *
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cpu'
enc_seq_len = 10
dec_seq_len = 4
output_sequence_length = 4

d_model = 512
d_ff=2048
lr = 0.005
epochs = 5
dropout=0.2
n_heads = 8

N=2
h=8
batch_size = 64
step_no=1
dimension=1


print('number of layer',N,'d_model',d_model,'d_ff',d_ff,'n_heads',h,'dropout',dropout)

train_dec_input=torch.load("train_dec_input.pt").unsqueeze(2)
train_encoder_input=torch.load('train_encoder_input.pt').unsqueeze(2)
train_output=torch.load('train_output.pt').unsqueeze(2)

test_encoder_input=torch.load('test_encoder_input.pt').unsqueeze(2)
test_output=torch.load('test_output.pt').unsqueeze(2)

validation_encoder_input=torch.load('validation_encoder_input.pt').unsqueeze(2)
validation_output=torch.load('validation_output.pt').unsqueeze(2)
validation_dec_input=torch.load('validation_dec_input.pt').unsqueeze(2)

T=make_model(enc_seq_len,dec_seq_len,output_sequence_length,N,d_model,d_ff,h,dropout)
T=T.to(device)
optimizer = torch.optim.Adam(T.parameters(),betas=(0.9, 0.98), eps=1e-09, lr=lr)

def adjust_optim(optimizer=optimizer, step_no=step_no, dim_val=d_model):
    learning_rate = (dim_val ** (0.5)) * min(step_no ** (0.5), step_no * 5000 ** (-1.5))
    print('learning_rate updated to:',learning_rate)
    optimizer.param_groups[0]['lr'] = learning_rate


best_loss=1000000
best_loss=1000000
round_batch=train_encoder_input.size(0)//batch_size
val_batch=Batch(validation_encoder_input,validation_dec_input)
for each in range(epochs):
    s=0
    e=batch_size
    T.train()
    running_loss = 0
    adjust_optim()
    step_no += 1

    for b in range(round_batch+1):
        optimizer.zero_grad()
        X_enc=train_encoder_input[s:e]
        X_dec=train_dec_input[s:e]
        X_enc=X_enc.to(device)
        X_dec = X_dec.to(device)
        Y=train_output[s:e]
        Y = Y.to(device)

        batch=Batch(X_enc,X_dec)
        s=e
        e=e+batch_size
        i=0
        #Forward pass and calculate loss
        net_out = T(batch.src,batch.trg,batch.src_mask,batch.trg_mask)
        net_out=net_out.to(device)
        loss_func=nn.MSELoss()
        #print(net_out.shape)
        loss = torch.sqrt(loss_func(net_out ,Y))
        loss.backward()
        running_loss += (loss**2)*batch.src.size(0)
        #print(loss)
        #backwards pass
        optimizer.step()
    T.eval()
    model_out = T(val_batch.src,val_batch.trg,val_batch.src_mask,val_batch.trg_mask)
    model_out = model_out.to(device)

    val_loss=RMSE_PC(model_out,validation_output)
    if best_loss>val_loss.data:
        torch.save(T, 'best_model.pt')
        best_loss=val_loss.data
        #test_encoder_input = torch.load('test_encoder_input.pt')
        #test_output = torch.load('test_output.pt')
        #RMSE_PC(test_encoder_input,t,test_output)
        state = {'epoch': each + 1, 'state_dict': T.state_dict(),
                 'optimizer': optimizer.state_dict(), 'step': step_no}
        torch.save(state,'checkpoint.pt')
    print('Epoch:',each+1,' Training_RMSE_Loss:',torch.sqrt(running_loss/(train_encoder_input.size(0))))
    print('Epoch:',each+1,' Validation_RMSE_Loss:',val_loss)




