#@title CMC model 
import torch
import torch.nn as nn
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()
        
        ##############################################################################
        #                               YOUR CODE HERE                               #
        ##############################################################################
        # TODO: Build an encoder with the architecture as specified above.           #
        ##############################################################################
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 12, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(48, 24, 4, 2, 1),
            nn.ReLU()
        ) 
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, x):
        h = self.encoder(x)
        return h
feat_dim = 24 * 4 * 4
class EncoderCMC(nn.Module):
    def __init__(self):
        super(EncoderCMC, self).__init__()

        self.l_encoder = Encoder(in_channels=1)
        self.ab_encoder = Encoder(in_channels=2)

    def forward(self, x):
        '''
        Extract features from L and ab channels.

        Args:
            x: torch.tensor

        Returns:
            feat_l: torch.tensor, (-1, feat_dim)
            feat_ab: torch.tensor, (-1, feat_dim)
        '''

        feat_l = self.l_encoder(x[:,0:1,:,:]).flatten(start_dim=1)
        feat_ab = self.ab_encoder(x[:,1:,:,:]).flatten(start_dim=1)
        return feat_l, feat_ab
class EncoderCMC_Cat(nn.Module):
    '''
    Wraper class for EncoderCMC. Concatenate feat_l and feat_ab to form a single 
    representation vector. This enables us to reuse the train_classifier routine.
    '''
    def __init__(self, model):
        super(EncoderCMC_Cat, self).__init__()
        self.model = model

    def forward(self, x):
        feat_l, feat_ab = self.model(x)
        return torch.cat((feat_l, feat_ab), dim=1)        