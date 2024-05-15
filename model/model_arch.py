import torch
from torch import nn

class OFFstateANN(nn.Module):
    
    def __init__(self, input_dim=5, output_dim=101):
        super(OFFstateANN, self).__init__()
        self.ANN = nn.Sequential(
            nn.Linear(input_dim, 16),   nn.Tanh()   ,
            nn.Linear(16, 32),          nn.Tanh()   ,
            nn.Linear(32, 64),          nn.PReLU()  ,
            nn.Linear(64, 96),          nn.PReLU()  ,
            nn.Linear(96, 140),         nn.PReLU()  ,
            nn.Linear(140, 154),
            nn.Linear(154, output_dim)
        )
    def forward(self, input_param):
        '''

        '''
        return self.ANN(input_param)
    
class ONstateANN(nn.Module):
    
    def __init__(self, input_dim=5, output_dim=101):
        super(ONstateANN, self).__init__()
        self.ANN = nn.Sequential(
            nn.Linear(input_dim, 16),   nn.Tanh()   ,
            nn.Linear(16, 32),          nn.PReLU()  ,
            nn.Linear(32, 64),          nn.Tanh()   ,
            nn.Linear(64, 96),          nn.PReLU()  ,
            nn.Linear(96, 140),         nn.Tanh()   ,
            nn.Linear(140, 154),
            nn.Linear(154, output_dim)
        )
    def transform()
    def forward(self, input_param):
        '''
    
        '''
        return self.ANN(input_param)
    
class TransitionANN(nn.Module):
    '''
    
    '''
    def __init__(self, input_dim=4, output_dim=1):
        super(TransitionANN, self).__init__()
        self.ANN = nn.Sequential(
            nn.Linear(input_dim, 16),   nn.Tanh()   ,
            nn.Linear(16, 32),          nn.PReLU()  ,
            nn.Linear(32, 64),          nn.Tanh()   ,
            nn.Linear(64, 96),          nn.PReLU()  ,
            nn.Linear(96, 140),         nn.Tanh()   ,
            nn.Linear(140, 154),        nn.PReLU()  ,
            nn.Linear(154, 180),        nn.PReLU()  ,
            nn.Linear(180, output_dim)
        )
    def forward(self, input_param):
        return self.ANN(input_param)
    
class SplitTransition(nn.Module):
    def __init__(self):
        super(SplitTransition, self).__init__()
        self.OFFstateANN = OFFstateANN()
        self.ONstateANN = ONstateANN()
        self.TransitionANN = TransitionANN()

    def forward(self, inputA, inputB):
        # Forward pass through Model A and B
        outputA = self.modelA(inputA)
        outputB = self.modelB(inputB)

        # Concatenate outputs and pass through Model C
        combined_output = torch.cat((outputA, outputB), dim=1)
        final_output = self.modelC(combined_output)
        return final_output
