# %matplotlib inline
import os, time, pickle, argparse
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import beta
torch.set_printoptions(threshold=10000)
np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='RSAutoML')
parser.add_argument('--Train_Method', type=str, default='AutoML', help='options: AutoML, Supervised')
parser.add_argument('--Val_Type', type=str, default='last_batch', help='options: last_batch, last_random')
parser.add_argument('--Loss_Type', type=str, default='MSE_sigmoid', help='options: MSE_sigmoid   MSE_no_sigmoid  BCEWithLogitsLoss   CrossEntropyLoss')
parser.add_argument('--Data_Set', type=str, default='ml-20m', help='options: ml-20m ml-latest')
parser.add_argument('--Dy_Emb_Num', type=int, default=2, help='options: 1, 2')
parser.add_argument('--random_seed', type=int, default=3000, help='options: 1, 2, ...')
args = parser.parse_args()

Model_Gpu  = torch.cuda.is_available()
device     = torch.device('cuda:0' if Model_Gpu else 'cpu')
DATA_PATH  = './data'
DATA_SET   = args.Data_Set
Batch_Size = 500     # batch size
LR_model   = 0.001   # learning rate
LR_darts   = 0.0001  # learning rate
Epoch      = 1       # train epoch
Beta_Beta  = 20      # beta for Beta distribution
H_alpha    = 0       # for nn.KLDivLoss 0.001

if DATA_SET == 'ml-20m':
    Train_Size   = 15000000      # training dataset size
elif DATA_SET == 'ml-latest':
    Train_Size = 22000000  # training dataset size

Test_Size    = 5000000       # training dataset size
Emb_Size     = [2, 4, 8, 16, 64, 128]  # 1,2,4,8,16,32,64,128,256,512
fixed_emb_size = sum(Emb_Size)

Val_Type     = args.Val_Type  # last_batch last_random
Dy_Emb_Num = args.Dy_Emb_Num
Loss_Type    = args.Loss_Type  # MSE_sigmoid   MSE_no_sigmoid  BCEWithLogitsLoss   CrossEntropyLoss

random_seed = args.random_seed
torch.manual_seed(random_seed)
if Model_Gpu:
    torch.cuda.manual_seed(random_seed)

print('\n****************************************************************************************\n')
print('os.getpid():   ', os.getpid())
if torch.cuda.is_available():
    print('torch.cuda:    ', torch.cuda.is_available(), torch.cuda.current_device(), torch.cuda.device_count(), torch.cuda.get_device_name(0), torch.cuda.device(torch.cuda.current_device()))
else:
    print('GPU is not available!!!')
print('Train_Size:    ', Train_Size)
print('Test_Size:     ', Test_Size)
print('fixed_emb_size:', fixed_emb_size)
print('Loss_Type:     ', Loss_Type)
print('Val_Type:      ', Val_Type)
print('Beta_Beta:     ', Beta_Beta)
print('H_alpha:       ', H_alpha)
print('LR_model:      ', LR_model)
print('LR_darts:      ', LR_darts)
print('\n****************************************************************************************\n')


def load_data():
    train_features, test_features, train_target, test_target \
        = pickle.load(open('{}/{}_TrainTest_{}_{}.data'.format(DATA_PATH, DATA_SET, Train_Size, Output_Dim), mode='rb'))
    test_features, test_target = test_features[:Test_Size], test_target[:Test_Size]
    genome_scores_dict = pickle.load(open('./{}/{}_GenomeScoresDict.data'.format(DATA_PATH, DATA_SET), mode='rb'))
    train_feature_data = pd.DataFrame(train_features, columns=['userId', 'movieId', 'user_frequency', 'movie_frequency'])
    test_feature_data = pd.DataFrame(test_features, columns=['userId', 'movieId', 'user_frequency', 'movie_frequency'])
    User_Num = max(train_feature_data['userId'].max() + 1, test_feature_data['userId'].max() + 1)  # 138494
    Movie_Num = max(train_feature_data['movieId'].max() + 1, test_feature_data['movieId'].max() + 1)  # 131263
    max_user_popularity = max(train_feature_data['user_frequency'].max()+1, test_feature_data['user_frequency'].max()+1)
    max_movie_popularity = max(train_feature_data['movie_frequency'].max() + 1, test_feature_data['movie_frequency'].max() + 1)
    # print('train_feature_data\n', train_feature_data)
    # print(train_feature_data.info())
    # print(train_feature_data.describe())
    return train_features, test_features, train_target, test_target, genome_scores_dict, \
           train_feature_data, test_feature_data, len(train_features), len(test_features), \
           User_Num, Movie_Num, max_user_popularity, max_movie_popularity


def Batch_Losses(Loss_Type, prediction, target):
    if Loss_Type == 'MSE_sigmoid':
        return nn.MSELoss(reduction='none')(nn.Sigmoid()(prediction), target)
    elif Loss_Type == 'MSE_no_sigmoid':
        return nn.MSELoss(reduction='none')(prediction, target)
    elif Loss_Type == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(reduction='none')(prediction, target)
    elif Loss_Type == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss(reduction='none')(prediction, target)
    else:
        print('No such Loss_Type.')


def Batch_Accuracies(Loss_Type, prediction, target):
    with torch.no_grad():
        if Loss_Type == 'MSE_sigmoid':
            predicted = 1 * (torch.sigmoid(prediction).data > 0.5)
        elif Loss_Type == 'MSE_no_sigmoid':
            predicted = 1 * (prediction > 0.5)
        elif Loss_Type == 'BCEWithLogitsLoss':
            predicted = 1 * (torch.sigmoid(prediction).data > 0.5)
        elif Loss_Type == 'CrossEntropyLoss':
            _, predicted = torch.max(prediction, 1)
        else:
            print('No such Loss_Type.')

        Batch_Accuracies = 1 * (predicted == target)
        Batch_Accuracies = list(Batch_Accuracies.detach().cpu().numpy())
        return Batch_Accuracies


def Beta(length, popularity, be=10):
    x = [i/length for i in range(length+1)]
    cdfs = [beta.cdf(x[i+1], popularity, be) - beta.cdf(x[i], popularity, be) for i in range(length)]
    return cdfs


class RS_MLP(nn.Module):
    def __init__(self, Output_Dim, Dynamic_Emb_Num):
        super(RS_MLP, self).__init__()
        self.emb_user = nn.Embedding(num_embeddings=User_Num, embedding_dim=fixed_emb_size)
        self.emb_movie = nn.Embedding(num_embeddings=Movie_Num, embedding_dim=fixed_emb_size)
        self.bn_user = nn.BatchNorm1d(fixed_emb_size)
        self.bn_movie = nn.BatchNorm1d(fixed_emb_size)
        self.tanh = nn.Tanh()
        self.movie_transfrom = nn.Sequential(  # nn.BatchNorm1d(1128),
            nn.Linear(1128, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Linear(512, fixed_emb_size))
        self.transfrom = nn.Sequential(
            nn.BatchNorm1d(fixed_emb_size * 2),
            nn.Linear(fixed_emb_size * 2, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Linear(512, Output_Dim))
        self.den = Dynamic_Emb_Num

    def forward(self, userId, movieId, movie_vec):
        user_emb = self.emb_user(userId)
        movie_emb = None if self.den == 1 else self.emb_movie(movieId)
        # v_user  = sum([torch.reshape(u_weight[:, i], (len(u_weight), -1)) * self.tanh(self.bn_user(self.W_user[i](user_emb[:,Emb_Split[i]:Emb_Split[i+1]]))) for i in range(len(Emb_Size))])
        # v_movie = sum([torch.reshape(m_weight[:, i], (len(m_weight), -1)) * self.tanh(self.bn_movie(self.W_movie[i](movie_emb[:,Emb_Split[i]:Emb_Split[i+1]]))) for i in range(len(Emb_Size))]) if self.den == 2 else self.movie_transfrom(movie_vec)

        v_user = user_emb
        v_movie = self.movie_transfrom(movie_vec) if self.den == 1 else movie_emb

        user_movie = torch.cat((v_user, v_movie), 1)
        return self.transfrom(user_movie)


def update_RS(index, features, Len_Features, target, mode):
    """ Update RS's embeddings and NN """
    global train_sample_loss, train_sample_accuracy
    index_end = index + Batch_Size
    if index_end >= Len_Features:
        batch_train = features[index:Len_Features]
        batch_train_target = target[index:Len_Features]
    else:
        batch_train = features[index:index_end]
        batch_train_target = target[index:index_end]

    userId = torch.tensor(batch_train[:, 0], requires_grad=False).to(device)
    movieId = torch.tensor(batch_train[:, 1], requires_grad=False).to(device)
    movie_vec = torch.tensor([genome_scores_dict[str(batch_train[:, 1][i])] for i in range(len(batch_train[:, 1]))],
                             requires_grad=False).to(device) if Dy_Emb_Num == 1 else None
    batch_train_target = torch.tensor(batch_train_target,
                                      dtype=torch.int64 if Loss_Type == 'CrossEntropyLoss' else torch.float32,
                                      requires_grad=False).to(device)

    rating = model(userId, movieId, movie_vec)
    rating = rating.squeeze(1).squeeze(1) if Loss_Type == 'CrossEntropyLoss' else rating.squeeze(1)

    batch_losses = Batch_Losses(Loss_Type, rating, batch_train_target)
    loss = sum(batch_losses)
    batch_accuracies = Batch_Accuracies(Loss_Type, rating, batch_train_target)
    # accuracy = sum(batch_accuracies) / len(batch_train_target)
    # print('loss3', loss, '\naccuracy', accuracy)

    train_sample_loss += list(batch_losses.detach().cpu().numpy())
    losses[mode].append(loss.detach().cpu().numpy())
    train_sample_accuracy += batch_accuracies
    accuracies[mode].append((sum(batch_accuracies), len(batch_train_target)))

    optimizer_model.zero_grad()
    loss.backward()
    optimizer_model.step()


if __name__ == "__main__":
    Output_Dim = 5 if Loss_Type == 'CrossEntropyLoss' else 1
    train_features, test_features, train_target, test_target, genome_scores_dict, \
    train_feature_data, test_feature_data, Len_Train_Features, Len_Test_Features, \
    User_Num, Movie_Num, max_user_popularity, max_movie_popularity = load_data()
    # Len_Train_Features, Len_Test_Features = 100000, 100000
    # Len_Train_Features = 10000
    train_feature_data, test_feature_data = train_feature_data[:Len_Train_Features], test_feature_data[:Len_Test_Features]

    model = RS_MLP(Output_Dim, Dy_Emb_Num)
    model.to(device)
    if Model_Gpu:
        print('\n========================================================================================\n')
        print('Memory:    ', torch.cuda.memory_allocated(0) / 1024 ** 3, 'GB', torch.cuda.memory_cached(0) / 1024 ** 3, 'GB')
        print('\n========================================================================================\n')

    t0 = time.time()

    optimizer_model = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_model, weight_decay=0)

    losses = {'train': [], 'test': []}
    accuracies = {'train': [], 'test': []}
    train_sample_loss = list()
    train_sample_accuracy = list()
    print('\n******************************************Train******************************************\n')
    for epoch_i in range(Epoch):
        #############################train#############################
        index = 0
        while index < Len_Train_Features:

            update_RS(index, train_features, Len_Train_Features, train_target, mode='train')

            if len(losses['train']) % 10 == 0:
                print('Epoch = {:>3}  Batch = {:>4}/{:>4} ({:.3f}%)    train_loss = {:.3f}     train_accuracy = {:.3f}     total_time = {:.3f} min'.format(
                    epoch_i, index + Batch_Size, Len_Train_Features, 100 * (index + Batch_Size) / Len_Train_Features, sum(losses['train'][-10:]) / 10,
                    sum([item[0] / item[1] for item in accuracies['train'][-10:]]) / 10,
                    (time.time() - t0) / 60))

            index += Batch_Size

    print('\n******************************************Test******************************************\n')
    t0 = time.time()
    index = 0
    while index < Len_Test_Features:

        update_RS(index, test_features, Len_Test_Features, test_target, mode='test')

        if len(losses['test']) % 10 == 0:
            print(
                'Test   Batch = {:>4}/{:>4} ({:.3f}%)     test_loss = {:.3f}     test_accuracy = {:.3f}     whole_time = {:.3f} min'.format(
                    index + Batch_Size, Len_Test_Features, 100 * (index + Batch_Size) / Len_Test_Features,
                    sum(losses['test'][-10:]) / 10,
                    sum([item[0] / item[1] for item in accuracies['test'][-10:]]) / 10, (time.time() - t0) / 60))

        index += Batch_Size

    correct_num = sum([item[0] for item in accuracies['test']])
    test_num = sum([item[1] for item in accuracies['test']])

    print('Test Loss: {:.4f}'.format(sum(losses['test']) / test_num))

    print('Test Correct Num: {}'.format(correct_num))
    print('Test Num: {}'.format(test_num))

    print('Test Accuracy: {:.4f}'.format(correct_num / test_num))

    # Save model
    save_model_name = './save_model/Fixed_DyEmbNum{}_LossType{}_TestAcc{:.4f}'.format(
        Dy_Emb_Num, Loss_Type,
        correct_num / test_num)
    torch.save(model.state_dict(), save_model_name + '.pt')
    print('Model saved to ' + save_model_name + '.pt')

    feature_data = pd.concat([train_feature_data, test_feature_data])

    print("feature_data: ", feature_data.shape[0], feature_data.shape[1])

    feature_data['loss_{}'.format(Emb_Size)] = pd.DataFrame([[i] for i in train_sample_loss])
    feature_data['acc_{}'.format(Emb_Size)] = pd.DataFrame([[i] for i in train_sample_accuracy])


    if Model_Gpu:
        print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        print('Memory:    ', torch.cuda.memory_allocated(0) / 1024 ** 3, 'GB', torch.cuda.memory_cached(0) / 1024 ** 3, 'GB')
        torch.cuda.empty_cache()
        print('Memory:    ', torch.cuda.memory_allocated(0) / 1024 ** 3, 'GB', torch.cuda.memory_cached(0) / 1024 ** 3, 'GB')
        print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')


    feature_data.to_csv('./results/feature_data_with_loss_{}_Fixed_{}_{}.csv'.format(Dy_Emb_Num, Loss_Type, DATA_SET), index=None)

    result_user, result_movie = [], []
    for i in range(1, 100):
        feature_data1 = feature_data[feature_data['user_frequency'] == i]
        result_user.append(list(feature_data1.mean(axis=0)) + [len(feature_data1)])
    Head = list(feature_data.columns) + ['count']
    pd.DataFrame(result_user).to_csv('./results/result_{}_Fixed_{}_{}_user.csv'.format(Dy_Emb_Num, Loss_Type, DATA_SET), index=None,
                                     header=Head)

    for i in range(1, 100):
        feature_data1 = feature_data[feature_data['movie_frequency'] == i]
        result_movie.append(list(feature_data1.mean(axis=0)) + [len(feature_data1)])
    Head = list(feature_data.columns) + ['count']
    pd.DataFrame(result_movie).to_csv('./results/result_{}_Fixed_{}_{}_movie.csv'.format(Dy_Emb_Num, Loss_Type, DATA_SET), index=None,
                                      header=Head)

    result = []
    for i in range(int(Train_Size / 1000000)):
        feature_data1 = feature_data[i * 1000000:(i + 1) * 1000000]
        result.append(list(feature_data1.mean(axis=0)) + [len(feature_data1)])

    Head = list(feature_data.columns) + ['count']
    pd.DataFrame(result).to_csv('./results/result_{}_Fixed_{}_{}_trendency.csv'.format(Dy_Emb_Num, Loss_Type, DATA_SET), index=None, header=Head)


    print('\n****************************************************************************************\n')
    print('os.getpid():   ', os.getpid())
    if torch.cuda.is_available():
        print('torch.cuda:    ', torch.cuda.is_available(), torch.cuda.current_device(), torch.cuda.device_count(), torch.cuda.get_device_name(0), torch.cuda.device(torch.cuda.current_device()))
    else:
        print('GPU is not available!!!')
    print('Train_Size:    ', Train_Size)
    print('Test_Size:     ', Test_Size)
    print('fixed_emb_size:', fixed_emb_size)
    print('Loss_Type:     ', Loss_Type)
    print('Val_Type:      ', Val_Type)
    print('Beta_Beta:     ', Beta_Beta)
    print('H_alpha:       ', H_alpha)
    print('LR_model:      ', LR_model)
    print('LR_darts:      ', LR_darts)
    print('\n****************************************************************************************\n')
    print('done')