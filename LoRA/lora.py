import torch 
import torch.nn as nn 

class LoRA(nn.Module):
    def __init__(self, k_in, d_out, r, alpha, W0):
        super(LoRA,self).__init__()
        
        # nn.Parameter so when training pytorch knows that A and B are trainable parameters.
        self.A = nn.Parameter(torch.randn(r, k_in,std = 0.02)) # std 0.02 because standard is 1, and it is big without reason in this case.
        self.B = nn.Parameter(torch.zeros(d_out, r))
        assert W0.shape == (d_out, k_in), "W0 dimensions don't match specified d_out and k_in"

        self.W0 = W0.requires_grad_(False) # Prevents W0 from participating in backpropagation, freezing it.

        assert (torch.matmul(self.B,self.A)).shape == self.W0.shape, "Error asserting W0 shape"
        self.r = r 
 
        if alpha is None:
            self.alpha = r # handles alpha/r, ensuring it is at least 1 if alpha is not defined.
        else:
            self.alpha = alpha
    def forward(self, x):
        scaler = self.alpha / self.r
        deltaW = scaler*torch.matmul(self.B,self.A)
        h = torch.matmul(self.W0,x) + torch.matmul(deltaW,x)
        return h
        


if __name__ == '__main__':
    test_lora = LoRA(10,12,20,11)
