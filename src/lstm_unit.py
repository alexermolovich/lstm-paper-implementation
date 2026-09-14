""" file containing the basic LSTM implementation to study the overall works of the internal part of the LSTM """ 


from cProfile import label
from enum import Enum

import numpy
from sympy import Array
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim import Adam
import lightning as L
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Parameter as t_Param
from torch import tensor as tt
from torch import normal as tn 
import numpy as np

class TrainingStagesLSTM(Enum):
    # Simple lifecycle markers for the training process.
    PENDING = 0,
    STARTED = 1,
    FINISHED = 2,  

class TrainingProgress:
    # Stores lightweight status information for the current training run.

    cur_batch_index : int 
    cur_training_status : TrainingStagesLSTM 
    description : str 
    
    def __init__(self):
        
        super.__init__()

        self.cur_batch_index = 0 
        self.cur_training_status = TrainingStagesLSTM.PENDING
        self.description = "Default"

class LSTMUnit(L.LightningModule):
    # Lightning module that holds the custom LSTM weights and training logic.
    training_stats: TrainingProgress 
  
    def init(self):
        super.__init__() 
          
        # Start with scalar tensors that are reused while building trainable weights.
        mean = torch.tensor(0.0)
        std  = torch.tensor(0.0)
        
        """ Basic paramter initialization for an LSTP unit """
        # Forget gate parameters decide how much previous long-term memory is kept.
        self.wlr1 = t_Param(tn(mean = mean, std = std), requires_grad = True) 
        self.wlr2 = t_Param(tn(mean = mean, std = std), requires_grad = True) 
        self.blr1 = t_Param(tn(0.), requires_grad = True) 
       
        # Input gate parameters control how much candidate memory is written.
        self.wpr1  = t_Param(tn(mean = mean, std = std), requires_grad = True) 
        self.wpr2 = t_Param(tn(mean = mean, std = std), requires_grad = True) 
        self.bpr1 = t_Param(tn(0.), requires_grad = True) 
         
        # Candidate memory parameters generate the new content for the cell state.
        self.wp1 = t_Param(tn(mean = mean, std = std), requires_grad = True) 
        self.wp2 = t_Param(tn(mean = mean, std = std), requires_grad = True) 
        self.bp1 = t_Param(tn(0.), requires_grad = True) 
        
        # Output gate parameters control what part of memory becomes the hidden state.
        self.wo1 = t_Param(tn(mean = mean, std = std), requires_grad = True) 
        self.wo2 = t_Param(tn(mean = mean, std = std), requires_grad = True) 
        self.bo1 = t_Param(tn(0.), requires_grad = True) 
   
        # Keep a small progress object so training state can be inspected later.
        self.training_stats = TrainingProgress()
    
    # Forward pass 
    def forward(self, training_input : numpy.array):
        # Begin each sequence with empty long-term and short-term memory values.
        long_term_memory = 0
        short_term_memory = 0 

        # Process the sequence one element at a time and keep the latest hidden state.
        for training_data_unit in training_input:
            long_term_memory, short_term_memory = self.lstm_unit(training_data_unit)   
        return short_term_memory  

    # Basic function implementation for a single LSTM unit pass
    def lstm_unit(self, input_value, long_memory, short_memory):
        # Forget gate: decide how much of the previous cell state should remain.
        long_remebered_percent = torch.sigmoid((short_memory * self.wlr1) + (input_value * self.wlr2) + self.blr1)
        
        # Input gate: decide how much new candidate information should be written.
        potential_remeber_percent = torch.sigmoid((short_memory * self.wpr1) + (input_value * self.wpr2) + self.bpr1) 
        # Candidate state: build a possible memory update from the input and hidden state.
        potential_memory = torch.tanh((short_memory * self.wp1) + (input_value * self.wp2) + self.bp1 ) 
        # Update the long-term memory by mixing the retained past with the new candidate.
        updated_long_memory = (( long_memory * long_remebered_percent ) + potential_remeber_percent * potential_memory) 
        # Output gate: choose how much of the updated cell state becomes visible output.
        output_percent = torch.sigmoid((short_memory * long_remebered_percent) + (potential_remeber_percent * potential_memory) + self.bo1)
        # The short-term memory is the exposed hidden state for the next time step.
        updated_short_memory = torch.tanh(updated_long_memory) + output_percent 
        
        return ([updated_long_memory, updated_short_memory])   
   
    def congfigure_optimizers(self):
        """Configuring optimizers"""
        
        # Adam optimizer
        adam_optimized = Adam(self.parameters())  
        return adam_optimized 

    # Calculates loss and training progress 
    def training_step(self, batch, batch_index):
        # Updating current training statistics 
        if self.training_stats.cur_training_status == TrainingStagesLSTM.PENDING: 
            # Save a readable status message and the current batch during the first pass.
            self.training_stats.description = "Training is running with the default LSTM configuration."
            self.training_stats.cur_batch_index = batch_index 
             
        # Split the batch into model inputs and expected targets.
        input_i, label_i = batch
        # Run the sequence through the model and collect its prediction.
        output_i = self.forward(input_i[0]) 
    
        # Use squared error to measure the gap between prediction and label.
        loss = (output_i - label_i) ** 2  
        # Record the loss together with the batch index for easier debugging.
        self.log(f"training_loss{loss}, on the following batch index {self.training_stats.cur_batch_index}") 
        
        return loss
