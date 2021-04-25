import lstm_encoder_decoder
import torch.nn as nn

import torch

def MAE(pred, true):
    return torch.mean(torch.abs(pred-true))

def MSE(pred, true):
    return torch.mean((pred-true)**2)


def RMSE_PC(model_out,test_output):
    loss_func = nn.MSELoss()
    mse=loss_func(model_out,test_output)
    RMSE_loss=torch.sqrt(mse)
    return RMSE_loss

Xtest=torch.load('test_encoder_input.pt').transpose(0,1).unsqueeze(2)
Ytest=torch.load('test_output.pt').transpose(0,1).unsqueeze(2)

print(Xtest.shape)
print(Ytest.shape)
model=torch.load('lstm_model.pt')
model.eval()

output=model.predict(Xtest,24)
print(output.shape)
mse=MSE(output,Ytest)
print(mse)
print(MAE(output,Ytest))