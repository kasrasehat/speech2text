

import os
import torch
import argparse
import pickle
from torch.optim.lr_scheduler import StepLR
from Data_preprocess import prepare_data
from torch.utils.data import TensorDataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC
from torch.optim.lr_scheduler import StepLR
import torch
from sklearn.model_selection import train_test_split
import numpy as np
#torch.cuda.empty_cache()

def train(args, model, device, train_loader, optimizer, epoch):

    model.train()
    model.freeze_feature_encoder()
    batch_loss = 0
    pp = 0
    tot_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        for i in data:

            path = 'speech2text/augmented_data/' + str(int(i)) + '.pkl'
            with open(path, 'rb') as fp:
                itemlist = pickle.load(fp)
            input_values = itemlist[0].to(device)
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

    print('Epoch: {}\tLoss: {:.6f}'.format(epoch, tot_loss / (len(train_loader.dataset))))
    return None

def evaluation(args, model, device, valid_loader, val_loss_min, epoch):

    model.eval()
    val_loss = 0
    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(valid_loader):

            for i in data:

                path = 'speech2text/augmented_data/' + str(int(i)) + '.pkl'
                with open(path, 'rb') as fp:
                    itemlist = pickle.load(fp)

                input_values = itemlist[0].to(device)
                loss = model(**input_values).loss
                val_loss += loss.item()


    val_loss = val_loss/len(valid_loader.dataset)
    print('\nValidation loss: {:.6f}\n'.format(val_loss))
    # save model if validation loss has decreased
    if val_loss < val_loss_min :

        if args.save_model:

            #filename = 'model_epock_{0}_val_loss_{1}.pt'.format(epoch, val_loss)
            #torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
            filename = 'E:/codes_py/speech2text/saved_models/hubert_epoch_{}'.format(epoch)
            torch.save(model.state_dict(),filename)
            val_loss_min = val_loss
        return val_loss_min

    else:
        return None


def main():
    # argparse = argparse.parse_args()
    parser = argparse.ArgumentParser(description='PyTorch speech2text')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--valid-batch-size', type=int, default=2000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.3, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
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
    model = model.to(device)
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    model.freeze_feature_encoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # weight_decay = 4e-4

    scheduler = StepLR(optimizer, step_size=15, gamma=args.gamma)

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

    files = os.listdir('speech2text/augmented_data')
    num_of_files = int(len(files) / 2)
    x = [i for i in range(num_of_files)]
    y = [i for i in range(num_of_files)]
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1, shuffle=True)

    tensor_x = torch.Tensor(x_train)  # transform to torch tensor
    tensor_y = torch.Tensor(y_train)
    train_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset

    tensor_x = torch.Tensor(x_valid)  # transform to torch tensor
    tensor_y = torch.Tensor(y_valid)
    valid_dataset = TensorDataset(tensor_x, tensor_y)

    train_loader = DataLoader(train_dataset, **kwargs_train)  # create your dataloader
    valid_loader = DataLoader(valid_dataset, **kwargs_val)

    val_loss_min = np.Inf
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()
        out_loss = evaluation(args, model, device, valid_loader, val_loss_min, epoch)
        if out_loss is not None:
            val_loss_min = out_loss



if __name__ == '__main__':
    main()

