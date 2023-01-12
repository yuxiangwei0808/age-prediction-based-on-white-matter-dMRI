from subprocess import check_output
import torch
import os
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch import nn
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from prior_loss import PrioriLoss
from harmodel import HarmodelEnsemble
from cutmix import cutmix_discrete
from torch.autograd import Variable
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.decomposition import PCA
import argparse
from einops import rearrange
import random
from tensorboardX import SummaryWriter


EPOCH = 300


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--data', type=str, default='./data/data_all_curated.npy')
    parser.add_argument('--channel', type=int, default=1024)
    parser.add_argument('--hemi', type=str, default=None)
    return parser.parse_args()


def save_model(epoch, try_idx, model, optimizer, pred_mae, r2):
    if not os.path.isdir('./checkpoint'):
        os.mkdir('./checkpoint')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, './checkpoint' + f'/Epoch-{epoch}-try-{try_idx}-MAE-{pred_mae}-R2-{r2}.pth')
    # with open('best_mae_baseline_tuning.txt', 'w') as f:
    #     f.write(f'{pred_mae.cpu().numpy()}')


def test(test_loader, model, epoch):
    total = torch.zeros(1).squeeze().cuda()
    mae = torch.zeros(1).squeeze().cuda()
    mse = torch.zeros(1).squeeze().cuda()
    pred_result = []
    true = []
    model.eval()
    with torch.no_grad():
        for data, label in test_loader:
            batch_size = data.shape[0]
            data, label = data.float().cuda(), label.float()
            output = model(data)
            output = output.cpu()

            pred_result.append(list(output.cpu().reshape(batch_size).numpy()))
            true.append(list(label.cpu().reshape(batch_size).numpy()))

            total += batch_size

            mae += torch.sum(torch.abs(label - output.squeeze()))
            mse += torch.sum(torch.pow(label - output.squeeze(), 2))

    pred_result = sum(pred_result, [])
    true = sum(true, [])
    pred_mae = mae / total
    pred_mse = mse / total
    r_2 = r2_score(true, pred_result)
    explained_var = explained_variance_score(true, pred_result)

    return pred_mae, pred_mse, r_2, explained_var


def train(writer, model, train_loader, test_loader, try_idx):
    best_mae = 20.0
    best_mse = 20.0
    best_r2 = -5
    best_evar = -5
    criterion = PrioriLoss(22, 37)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)

    for epoch in range(EPOCH):
        total = 0
        mae = torch.zeros(1).squeeze().cuda()
        model.train()
        for batch_idx, (data, label) in enumerate(train_loader):
            batch_size = data.shape[0]

            total += batch_size
            data, label = data.float().cuda(), label.cuda()
            optimizer.zero_grad()

            data = cutmix_discrete(data)
            output = model(data)

            mae += torch.sum(torch.abs(label - output.squeeze()))
            
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            scheduler.step()
        train_mae = mae / total
        pred_mae, pred_mse, r_2, explained_var = test(test_loader, model, epoch)
        # writer.add_scalars('MAE', {'train_mae': train_mae, 'test_mae': pred_mae}, epoch)
        if pred_mae < best_mae:
            best_mae = pred_mae
            if best_mae < 2.8:
                save_model(epoch, try_idx, model, optimizer, best_mae, best_r2)
            print(f"\nBest MAE:{pred_mae} at epoch {epoch}")
        if pred_mse < best_mse:
            best_mse = pred_mse
            print(f'Best MSE:{pred_mse} at epoch {epoch}')
        if r_2 > best_r2:
            best_r2 = r_2
            print(f'Best R_2:{best_r2} at epoch {epoch}')
        if explained_var > best_evar:
            best_evar = explained_var
            print(f'Best explained variance:{best_evar} at epoch {epoch}')
    return best_mae, best_mse, best_r2, best_evar, model, optimizer, scheduler


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）

    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = np.concatenate((X_train, X_part), axis=0)
            y_train = np.concatenate((y_train, y_part), axis=0)
    return X_train, y_train, X_valid, y_valid


def normalize(input):
    x = rearrange(input, 'b c l -> b l c')
    x = torch.tensor(x)
    output = None
    for i in range(x.shape[0]):
        temp = F.normalize(x[i], p=2, dim=0)
        if output is None:
            output = temp[None]
        else:
            output = torch.cat((output, temp[None]), dim=0)
    return np.array(output)


def main():
    args = get_args()
    device = torch.device('cuda')
    torch.cuda.set_device(args.cuda_id)

    if args.data:
        trainxx = np.load(args.data)

    features = [0, 1, 8, 9, 10, 11, 13, 14, 15, 16, 19, 26, 27, 28, 29, 32, 33, 34, 35]
    trainx = np.zeros(1)
    for k in range(trainxx.shape[0]):
        temp = np.zeros(1)
        for i in features:
            feature = trainxx[k, i, :][None]
            if temp.any():
                temp = np.concatenate((temp, feature), axis=0)  # x(n,num_cls)
            else:
                temp = feature
        if trainx.any():
            trainx = np.concatenate((trainx, temp[None]), axis=0)
        else:
            trainx = temp[None]

    results = []
    for _ in range(4):
        results.append([])

    trainx = normalize(trainx)

    if args.hemi:
        middle_idx = int(trainx.shape[1] / 2)
        if args.hemi == 'left':
            trainx = trainx[:, :middle_idx, :]
        elif args.hemi == 'right':
            trainx = trainx[:, middle_idx:, :]

    if args.num_cluster:
        list = [i for i in range(0, trainx.shape[1])]
        idx = random.sample(list, args.num_cluster)
        trainx = trainx[:, idx, :]

    trainy = np.load('./data/HCP_label.npy')  # (965, 8)
    trainy = trainy[:, 0]

    k = 5
    for i in range(k):
        # writer = SummaryWriter(f'log_{i}')
        writer = None

        x_train, y_train, x_test, y_test = get_k_fold_data(k, i, trainx, trainy)

        class TrainSet(Dataset):
            def __init__(self):
                self.x_train = x_train
                self.y_train = y_train

            def __getitem__(self, index):
                train_data = torch.tensor(self.x_train[index])
                train_label = torch.tensor(self.y_train[index])
                return train_data, train_label

            def __len__(self):
                return self.x_train.shape[0]

        class TestSet(Dataset):
            def __init__(self):
                self.x_test = x_test
                self.y_test = y_test

            def __getitem__(self, index):
                test_data = torch.tensor(self.x_test[index])
                test_label = torch.tensor(self.y_test[index])
                return test_data, test_label

            def __len__(self):
                return self.x_test.shape[0]

        set1 = TrainSet()
        set2 = TestSet()

        batch_size = 32
        train_loader = DataLoader(dataset=set1, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=set2, batch_size=batch_size, shuffle=True)

        model = HarmodelEnsemble(in_dim=trainx.shape[1], feature_len=trainx.shape[2], channel=args.channel).to(device)

        mae, mse, r2, var, model, optimizer, scheduler = train(writer, model, train_loader, test_loader, try_idx=i)

        print(f'results of this try: MAE:{mae}, MSE:{mse}, R_2:{r2}, E-Var:{var}')
        results[0].append(mae.cpu().float())
        results[1].append(mse.cpu().float())
        results[2].append(r2)
        results[3].append(var)

        # writer.export_scalars_to_json(f'./all_scalars_{i}.json')
    print(f'ave_mae:{np.mean(results[0])}, ave_mse:{np.mean(results[1])}, ave_r2:{np.mean(results[2])}, '
          f'ave_var:{np.mean(results[3])}')

    data_name = args.data.rstrip('.npy')[7:]
    idx = data_name.find('/')
    WM_region = data_name[:idx]
    if args.hemi:
        data_name = f'{data_name[idx+1:]}_{args.hemi}'
    with open(f'./results_{WM_region}_hemi_finest_{args.channel}channels.txt', 'a') as f:
        f.write(
            f'\nResults of {data_name} with {args.channel} channels:(ave_mae:{np.mean(results[0])}, ave_mse:{np.mean(results[1])}, ave_r2:{np.mean(results[2])}, '
            f'ave_var:{np.mean(results[3])})')


if __name__ == '__main__':
    main()
