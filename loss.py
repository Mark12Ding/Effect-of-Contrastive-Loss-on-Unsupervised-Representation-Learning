#@title Spring Like
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class CMCScore_spring(nn.Module):
    '''
    Calculate ℎ𝜃({𝑣1,𝑣2}) and ℎ𝜃({𝑣2,𝑣1}).

    To efficiently compute the scores, we use memories to store L and ab 
    representations. For more details on this, please refer to the original
    CMC paper
    '''

    def __init__(self, feat_dim, N, K, T=0.1, momentum=0.5):
        '''
        Args:
            feat_dim: int, dimension of the extracted representations
            N: int, number of samples in the dataset.
            K: int, number of negative examples.
            T: float, temeprature.
            momentum: float. momentum of memory. 
        '''
        super(CMCScore_spring, self).__init__()
        self.N = N
        self.K = K
        self.feat_dim = feat_dim
        self.ones = torch.ones(N).cuda()
        self.eps = 1e-7
        self.m = 1

        self.register_buffer('params', torch.tensor([K, T, momentum]))
        stdv = 1. / math.sqrt(feat_dim / 3)
        self.register_buffer('memory_l', torch.rand(N, feat_dim).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(N, feat_dim).mul_(2 * stdv).add_(-stdv))

    def forward(self, l, ab, y, idx=None):
        '''
        Args:
            l: torch.tensor. l channel representation. (-1, feat_dim)
            ab: torch.tensor. ab channel representation. (-1, feat_dim)
            y: torch.tensor. Dataset index corresponding to the input images.
            
        Returns:
            out_l: torch.tensor. (-1, K+1, 1)
            out_ab: torch.tensor. (-1, K+1, 1)
        '''
        K = int(self.params[0].item())
        T = self.params[1].item()

        momentum = self.params[2].item()
        batch_size = l.size(0)
        N = self.N
        feat_dim = self.feat_dim

        # normalize l and ab representations
        l = l / (l.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps)
        ab = ab / (ab.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps)

        # Randomly sample K indicies for each anchor data to form negative pairs. 
        if idx is None:
            idx = torch.multinomial(self.ones, batch_size*(self.K+1), replacement=True)
            idx = idx.view(batch_size, -1) # (N, K+1)
            # set the 0-th element to be positive sample.
            idx.select(1, 0).copy_(y.data)

            # idx[:,0]
        ##############################################################################
        # Compute out_l and out_ab. out_l is the normalized dot product between      # 
        # anchor l channel and randomly sampled ab channels. out_ab is the opposite. #
        # Using the stored representations from the memory to avoid computing new    #
        # representations on-the-fly. Make sure you use the idx varaible we created  #
        # above to retrieve representations from the memory.                         #
        ##############################################################################
        # normalized dot product for l
        weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
        weight_ab = weight_ab.view(batch_size, K + 1, feat_dim)
        out_l = weight_ab - l.view(batch_size, 1, feat_dim)
        out_l = out_l.norm(dim=-1,keepdim=True)
        out_l[:,1:] = F.relu(self.m - out_l[:,1:])
        
        ##############################################################################
        #                                  YOUR CODE HERE                            #
        ##############################################################################
        # normalized dot product for ab
        weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
        weight_l = weight_l.view(batch_size, K + 1, feat_dim)
        out_ab = weight_l - ab.view(batch_size, 1, feat_dim)
        out_ab = out_ab.norm(dim=-1,keepdim=True)
        out_ab[:,1:] = F.relu(self.m - out_ab[:,1:])
        ##############################################################################
        #                                  IMPORTANT                                 #
        ##############################################################################
        out_l = out_l.contiguous()
        out_ab = out_ab.contiguous()

        # # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)

        return out_l, out_ab


class CMCScore_triplet(nn.Module):
    def __init__(self, feat_dim, N, K, T=0.1, momentum=0.5):
        '''
        Args:
            feat_dim: int, dimension of the extracted representations
            N: int, number of samples in the dataset.
            K: int, number of negative examples.
            T: float, temeprature.
            momentum: float. momentum of memory. 
        '''
        super(CMCScore_triplet, self).__init__()
        self.N = N
        self.K = K
        self.feat_dim = feat_dim
        self.ones = torch.ones(N).cuda()
        self.eps = 1e-7
        self.m = 1

        self.register_buffer('params', torch.tensor([K, T, momentum]))
        stdv = 1. / math.sqrt(feat_dim / 3)
        self.register_buffer('memory_l', torch.rand(N, feat_dim).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(N, feat_dim).mul_(2 * stdv).add_(-stdv))

    def forward(self, l, ab, y, idx=None):
        '''
        Args:
            l: torch.tensor. l channel representation. (-1, feat_dim)
            ab: torch.tensor. ab channel representation. (-1, feat_dim)
            y: torch.tensor. Dataset index corresponding to the input images.
            
        Returns:
            out_l: torch.tensor. (-1, K+1, 1)
            out_ab: torch.tensor. (-1, K+1, 1)
        '''
        K = int(self.params[0].item())
        T = self.params[1].item()

        momentum = self.params[2].item()
        batch_size = l.size(0)
        N = self.N
        feat_dim = self.feat_dim

        # normalize l and ab representations
        l = l / (l.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps)
        ab = ab / (ab.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps)

        # Randomly sample K indicies for each anchor data to form negative pairs. 
        if idx is None:
            idx = torch.multinomial(self.ones, batch_size*(self.K+1), replacement=True)
            idx = idx.view(batch_size, -1) # (N, K+1)
            # set the 0-th element to be positive sample.
            idx.select(1, 0).copy_(y.data)

            # idx[:,0]
       ##############################################################################
        # Compute out_l and out_ab. out_l is the normalized dot product between      # 
        # anchor l channel and randomly sampled ab channels. out_ab is the opposite. #
        # Using the stored representations from the memory to avoid computing new    #
        # representations on-the-fly. Make sure you use the idx varaible we created  #
        # above to retrieve representations from the memory.                         #
        ##############################################################################
        # normalized dot product for l
        weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
        weight_ab = weight_ab.view(batch_size, K + 1, feat_dim)
        pos_dis = (weight_ab[:, 0, :] -  l.view(batch_size, feat_dim)).norm(dim=-1,keepdim=True)
        out_l = weight_ab - l.view(batch_size, 1, feat_dim)
        out_l = out_l.norm(dim=-1,keepdim=True)
        out_l = -out_l + pos_dis.view(batch_size, 1, 1)
        out_l = F.relu(1 + out_l)
        
        ##############################################################################
        #                                  YOUR CODE HERE                            #
        ##############################################################################
        # normalized dot product for ab
        weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
        weight_l = weight_l.view(batch_size, K + 1, feat_dim)
        pos_dis = (weight_l[:, 0, :] - ab.view(batch_size, feat_dim)).norm(dim=-1,keepdim=True)
        out_ab = weight_l - ab.view(batch_size, 1, feat_dim)
        out_ab = out_ab.norm(dim=-1,keepdim=True)
        out_ab =  -out_ab + pos_dis.view(batch_size, 1, 1)
        out_ab = F.relu(1 + out_ab)
        
        ##############################################################################
        #                                  IMPORTANT                                 #
        ##############################################################################
        out_l = out_l.contiguous()
        out_ab = out_ab.contiguous()

        # # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)

        return out_l, out_ab

#@title softmax loss
class CMCScore_softmax(nn.Module):
    def __init__(self, feat_dim, N, K, T=0.1, momentum=0.5):
        super(CMCScore_softmax, self).__init__()
        self.N = N
        self.K = K
        self.feat_dim = feat_dim
        self.ones = torch.ones(N).cuda()
        self.eps = 1e-7

        self.register_buffer('params', torch.tensor([K, T, momentum]))
        stdv = 1. / math.sqrt(feat_dim / 3)
        self.register_buffer('memory_l', torch.rand(N, feat_dim).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(N, feat_dim).mul_(2 * stdv).add_(-stdv))

    def forward(self, l, ab, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()

        momentum = self.params[2].item()
        batch_size = l.size(0)
        N = self.N
        feat_dim = self.feat_dim

        # normalize l and ab representations
        l = l / (l.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps)
        ab = ab / (ab.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps)

        # Randomly sample K indicies for each anchor data to form negative pairs. 
        if idx is None:
            idx = torch.multinomial(self.ones, batch_size*(self.K+1), replacement=True)
            idx = idx.view(batch_size, -1) # (N, K+1)
            # set the 0-th element to be positive sample.
            idx.select(1, 0).copy_(y.data)
        weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
        weight_ab = weight_ab.view(batch_size, K + 1, feat_dim)
        out_l = torch.bmm(weight_ab, l.view(batch_size, feat_dim, 1))
        out_l = torch.div(out_l, T)

      
        # normalized dot product for ab
        weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
        weight_l = weight_l.view(batch_size, K + 1, feat_dim)
        out_ab = torch.bmm(weight_l, ab.view(batch_size, feat_dim, 1))
        out_ab = torch.div(out_ab, T)
        ##############################################################################
        #                                  IMPORTANT                                 #
        ##############################################################################
        out_l = out_l.contiguous()
        out_ab = out_ab.contiguous()

        # # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)

        return out_l, out_ab

class CMCScore_infonce(nn.Module):
    def __init__(self, feat_dim, N, K, T=0.5, momentum=0.5):
        super(CMCScore, self).__init__()
        self.N = N
        self.K = K
        self.feat_dim = feat_dim
        self.ones = torch.ones(N).cuda()
        self.eps = 1e-7

        self.register_buffer('params', torch.tensor([K, T, momentum]))
        stdv = 1. / math.sqrt(feat_dim / 3)
        self.register_buffer('memory_l', torch.rand(N, feat_dim).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(N, feat_dim).mul_(2 * stdv).add_(-stdv))

    def forward(self, l, ab, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        outputSize = self.memory_l.size(0)

        momentum = self.params[2].item()
        batch_size = l.size(0)
        N = self.N
        feat_dim = self.feat_dim

        # normalize l and ab representations
        l = l / (l.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps)
        ab = ab / (ab.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps)

        # Randomly sample K indicies for each anchor data to form negative pairs. 
        if idx is None:
            idx = torch.multinomial(self.ones, batch_size*(K+1), replacement=True)
            idx = idx.view(batch_size, -1) # (N, K+1)
            # set the 0-th element to be positive sample.
            idx.select(1, 0).copy_(y.data)

            # idx[:,0]
        ##############################################################################
        # Compute out_l and out_ab. out_l is the normalized dot product between      # 
        # anchor l channel and randomly sampled ab channels. out_ab is the opposite. #
        # Using the stored representations from the memory to avoid computing new    #
        # representations on-the-fly. Make sure you use the idx varaible we created  #
        # above to retrieve representations from the memory.                         #
        ##############################################################################
        # normalized dot product for l
        self.m = K
        weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
        weight_ab = weight_ab.view(batch_size, K + 1, feat_dim)
        P_l = torch.bmm(weight_ab, l.view(batch_size, feat_dim, 1))
        P_l = torch.div(P_l, T)
        P_l = torch.exp(P_l)
        Z_l = P_l.mean()*outputSize
        out_l = torch.div(P_l, Z_l) 
        # normalized dot product for ab
        weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
        weight_l = weight_l.view(batch_size, K + 1, feat_dim)
        P_ab = torch.bmm(weight_l, ab.view(batch_size, feat_dim, 1))
        P_ab = torch.div(P_ab, T)
        P_ab = torch.exp(P_ab)
        Z_ab = P_ab.mean()*outputSize
        out_ab = torch.div(P_ab, Z_ab) # P_ab = P_ab / (P_ab.mean(dim=1, keepdim=True)*N) # (N, N, 1)
  
        out_l = out_l.contiguous()
        out_ab = out_ab.contiguous()

        # # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)

        return out_l, out_ab
class Loss(nn.Module):
    """
    Softmax cross-entropy loss.
    """
    def __init__(self, softmax=False):
        super(Loss, self).__init__()
        self.softmax = softmax

    def forward(self, x):
        batch_size = x.shape[0]
        if self.softmax:
            x = x.squeeze()
            label = torch.zeros(batch_size, device=x.device, dtype=torch.long)
            loss = F.cross_entropy(x, label)
        else:
            loss = x.sum()/batch_size
        return loss


class NCECriterion(nn.Module):
    def __init__(self, n_data):
        super(NCECriterion, self).__init__()
        self.n_data = n_data
    def forward(self, x):
        eps = 1e-7
        bsz = x.shape[0]
        m = x.size(1) - 1
        Pn = 1 / float(self.n_data)
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()
        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz
        return loss
def align_loss(x,y,alpha=2):
  #xy postive pair
  return (x-y).norm(p=2, dim=1).pow(alpha).mean()
def uniform_loss(x, t=2):
  return torch.pdist(x,p=2).pow(2).mul(-t).exp().mean().log()





