
import numpy as np
import os
import torch
import argparse
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from Data_preprocess import prepare_data
from torch.utils.data import TensorDataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC
from torch.optim.lr_scheduler import StepLR
import torch

import numpy as np


# main
# test data is extracted in this section
# in this section data normalized with min_max method

def train(args, model, device, train_loader, optimizer, epoch, train_data):

    model.train()
    model.freeze_feature_encoder()
    batch_loss = 0
    pp = 0
    tot_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        loss_tot = 0
        for i in range(len(data)):

            input_values = train_data[int(data[i])]
            loss = model(**input_values).loss
            batch_loss += loss


        batch_loss_mean = batch_loss/len(data)
        optimizer.zero_grad()
        batch_loss_mean.backward()
        optimizer.step()
        tot_loss += batch_loss.item()
        batch_loss = 0

        if batch_idx % args.log_interval == 1 :
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx+1) / len(train_loader),
                       tot_loss / ((batch_idx + 1) * len(data))))
            filename = 'E:/codes_py/speech2text/saved_models/model_epoch_{}_progress_{}%'.format(epoch, 100. * np.ceil(batch_idx+1) / len(train_loader))
            torch.save(model.state_dict(), filename)

    print('Epoch: {}\tLoss: {:.6f}'.format(epoch, tot_loss / (len(train_loader.dataset))))
    filename = 'E:/codes_py/speech2text/saved_models/hubert_epoch_{}'.format(epoch)
    torch.save(model.state_dict(), filename)

    return tot_loss / (len(train_loader.dataset))


def main():
    # argparse = argparse.parse_args()
    parser = argparse.ArgumentParser(description='PyTorch finance EURUSD')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--valid-batch-size', type=int, default=2000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=45, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--weight', default=False,
                        help='path of pretrain weights')
    parser.add_argument('--resume', default=False,
                        help='path of resume weights , "./cnn_83.pt" OR "./FC_83.pt" OR False ')

    args = parser.parse_args()
    use_cuda = args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)

    kwargs_train = {'batch_size': args.batch_size}
    kwargs_train.update({'num_workers': 1,
                         'shuffle': True,
                         'drop_last': True},
                        )
    kwargs_val = {'batch_size': args.valid_batch_size}
    kwargs_val.update({'num_workers': 1,
                       'shuffle': True})


    # model = Net(input_dim=1, hidden_dim=30, layer_dim=1, output_dim=pred_len, dropout_prob=0, device= device).to(device)
    model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=4e-4)
    # weight_decay = 4e-4

    scheduler = StepLR(optimizer, step_size=6, gamma=args.gamma)

    if args.weight:
        if os.path.isfile(args.weight):
            checkpoint = torch.load(args.weight)
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                model.load_state_dict(checkpoint)

    # args.resume = False
    if args.resume:
        if os.path.isfile(args.resume):
            # checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            checkpoint = torch.load(args.resume)
            try:
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                model.load_state_dict(checkpoint)



    a1 = prepare_data()
    train_data, bug = a1.prep_data(
        path = "speech2text/pharmacy6.pickle")

    indices = []
    for i in range(len(train_data)):
        indices.append(i)


    ##train_data = train_data2
    tensor_x = torch.Tensor(indices)  # transform to torch tensor
    tensor_y = torch.Tensor(indices)
    train_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset


    train_loader = DataLoader(train_dataset, **kwargs_train)  # create your dataloader

    val_loss_min = np.Inf
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, train_data=train_data)
        scheduler.step()



if __name__ == '__main__':
    main()

