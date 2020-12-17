import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_cmc(model, contrast, criterion_l, criterion_ab, dataloader, epochs=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    patience = 3
    for epoch in range(epochs):
    
        model.train()
        ab_prob_epoch = 0
        l_prob_epoch = 0
        loss_epoch = 0
        last_loss = float('inf')
        for idx, (inputs, _, index) in enumerate(dataloader):

            batch_size = inputs.size(0)
            inputs = inputs.float().to(device)
            index = index.to(device)

            ##############################################################################
            #                               YOUR CODE HERE                               #
            ##############################################################################
            # TODO: compute loss values. Note that you need to provide label to          # 
            # CrossEntropyLoss. (Hint: the positive sample is always the first one).     #
            ##############################################################################
            feat_l, feat_ab = model(inputs)
            out_l, out_ab = contrast(feat_l, feat_ab, index)
            loss = criterion_l(out_l) + criterion_ab(out_ab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ##############################################################################
            #                               END OF YOUR CODE                             #
            ##############################################################################

            l_prob = F.softmax(out_l, dim=1)[:, 0, 0].sum().detach().cpu()
            ab_prob = F.softmax(out_ab, dim=1)[:, 0, 0].sum().detach().cpu()

            l_prob_epoch += l_prob
            ab_prob_epoch += ab_prob
            loss_epoch += loss.data


        l_prob_epoch = l_prob_epoch / len(dataloader.dataset)
        ab_prob_epoch = ab_prob_epoch / len(dataloader.dataset)
        loss_epoch = loss_epoch / len(dataloader)
        if last_loss > loss_epoch:
          patience = 3  
        else:
          patience -= 1
          if patience == 0:
            break  
        last_loss = loss_epoch.detach().cpu()  
        print('Epoch {}, loss {:.3f}\t' 
              'avg prob for L channel {:.3f}\t' 
              'avg prob for ab channels {:.3f}'.format(
                  epoch, loss_epoch.item(), l_prob_epoch.item(), ab_prob_epoch.item()))

    return model

def train_classfier(encoder, cls, dataloader, epochs=100, supervised=False):
    '''
    Args:
        encoder: trained/untrained encoder for unsupervised/supervised training.
        cls: linear classifier.
        dataloader: train partition.
        supervised: 

    Return:
        cls: linear clssifier.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(cls.parameters(), lr=0.001, weight_decay=1e-4)
    if supervised:
        optimizer = optim.Adam(list(cls.parameters())+list(encoder.parameters()), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    loss_traj = []
    accuracy_traj = []

    for epoch in range(epochs):

        loss_epoch = 0
        corrects_epoch = 0
        for x, y in dataloader:

            batch_size = x.size(0)
            x = x.float()
            ##############################################################################
            #                               YOUR CODE HERE                               #
            ##############################################################################
            # TODO: update the parameters of the classifer. If in supervised mode, the   #
            # parameter of the encoder is also updated.                                  #
            ##############################################################################
            x = x.to(device)
            y = y.to(device)
            outs = cls(encoder(x).flatten(start_dim=1))
            loss = criterion(outs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ##############################################################################
            #                               END OF YOUR CODE                             #
            ##############################################################################
            _, preds = torch.max(outs, 1)
            corrects_epoch += torch.sum(preds == y.data)
            loss_epoch += loss.detach()

        loss_traj.append(loss_epoch)
        epoch_acc = corrects_epoch.double() / len(dataloader.dataset)
        accuracy_traj.append(epoch_acc)
    
        if epoch % 10 == 0:
            print('Epoch {}, loss {:.3f}, train accuracy {}'.format(epoch, loss_epoch.item(), epoch_acc.item()))

    return cls, loss_traj
    
def test(encoder, cls, dataloader):
    '''
    Calculate the accuracy of the trained linear classifier on the test set.
    '''
    cls.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_epoch = 0
    corrects_epoch = 0
    for x, y in dataloader:
        
        x = x.float()
        batch_size = x.size(0)
        x, y = x.to(device), y.to(device)
        h = encoder(x).view(batch_size, -1)
        outs = cls(h)
        _, preds = torch.max(outs, 1)
        corrects_epoch += torch.sum(preds == y.data)

    epoch_acc = corrects_epoch.double() / len(dataloader.dataset)
    print('Test accuracy {}'.format(epoch_acc.item()))
