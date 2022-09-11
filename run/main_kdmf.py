import os
import json
import tqdm
import copy
import torch
import configparser
from torch import nn, optim

# import evaluation
from utility import evaluate
from utility import data
from model.KnowledgeMF import KnowledgeMF
from model.KDMLP import KDMLP
from utility.train_util import generate_result_path, explain_result_path, save_train_result, get_comparison_configlist


def train(k_model, k_opt, k_criterion, k_train_loader, k_test_loader, y_model, y_opt, y_criterion, train_loader, validation_loader, test_loader, config):
    yepoch_data = []
    init_hit_ratio, init_ndcg = evaluate.metrics(y_model, validation_loader, config.getint('EVALUATION', 'top_k'))
    yepoch_data.append({'epoch': -1, 'loss': '', 'HR': init_hit_ratio, 'NDCG': init_ndcg})
    kepoch_data = []
    init_mae, init_mse, init_rmse, init_r2 = evaluate.klinking_evaluate(k_model, k_test_loader)
    kepoch_data.append({'epoch': -1, 'loss': '', 'MAE': float(init_mae), 'MSE': float(init_mse), 'RMSE': float(init_rmse), 'R2': float(init_r2)})

    best_ymodel_state_dict = None
    best_kmodel_state_dict = None
    best_ndcg = 0
    best_r2 = -100000
    best_yepoch = 0
    best_kepoch = 0
    for epoch in range(config.getint('MODEL', 'epoch')):
        # Train K
        k_model.embed_k_GMF.weight.data.copy_(y_model.embed_k_MLP.weight)
        k_model.train()
        k_total_loss = 0
        for batch in k_train_loader:
            fromks, toks, scores = batch[0], batch[1], batch[2].float()
            fromks = fromks.to('cuda:0')
            toks = toks.to('cuda:0')
            scores = scores.to('cuda:0')
            # setting
            k_opt.zero_grad()
            pred = k_model(fromks, toks)
            k_loss = k_criterion(pred.view(-1), scores)
            k_loss.backward()
            k_opt.step()
            k_total_loss += k_loss.item()
        mae, mse, rmse, r2 = evaluate.klinking_evaluate(k_model, k_test_loader)
        kepoch_data.append({'epoch': epoch, 'loss': k_total_loss, 'MAE': float(mae), 'MSE': float(mse), 'RMSE': float(rmse), 'R2': float(r2)})
        if r2 > best_r2:
            best_r2 = r2
            best_kepoch = epoch
            best_kmodel_state_dict = copy.deepcopy(k_model.state_dict())
        print('>> [KTraining] <Epoch {}> Loss = {:.2f}, MAR = {:.4f}, MSE = {:.4f}, RMSE = {:.4f}, R2 = {:.4f}'.format(epoch, k_total_loss, mae, mse, rmse, r2))
        
        # Train Y
        y_model.embed_k_MLP.weight.data.copy_(k_model.embed_k_GMF.weight)
        y_model.train()
        y_total_loss = 0
        for batch in train_loader:
            users, items, ratings, user_priork, user_targetk, item_containk = batch[0], batch[1], batch[2].float(), batch[3], batch[4], batch[5] # tensor
            valid_priork_num, valid_targetk_num, valid_itemk_num = batch[6], batch[7], batch[8] # list
            users = users.to('cuda:0')
            items = items.to('cuda:0')
            ratings = ratings.to('cuda:0')

            # setting
            y_opt.zero_grad()
            pred = y_model(users, items, user_priork, user_targetk, item_containk, valid_priork_num, valid_targetk_num, valid_itemk_num)
            y_loss = y_criterion(pred.view(-1), ratings)
            y_loss.backward()
            y_opt.step()
            y_total_loss += y_loss.item()
        hit_ratio, ndcg = evaluate.metrics(y_model, validation_loader, config.getint('EVALUATION', 'top_k'))
        yepoch_data.append({'epoch': epoch, 'loss': y_total_loss, 'HR': hit_ratio, 'NDCG': ndcg})
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_yepoch = epoch
            best_ymodel_state_dict = copy.deepcopy(y_model.state_dict())
        print('>> [RSTraining]<Epoch {}> Loss = {:.2f}, HR = {:.4f}, NDCG = {:.4f}'.format(epoch, y_total_loss, hit_ratio, ndcg))
    
    print('>> [Training] the best knowledge epoch is {}'.format(best_kepoch))
    k_model.load_state_dict(best_kmodel_state_dict)
    final_mae, final_mse, final_rmse, final_r2 = evaluate.klinking_evaluate(k_model, k_test_loader)
    print('>> [Training] the best rs epoch is {}'.format(best_yepoch))
    y_model.load_state_dict(best_ymodel_state_dict)
    final_hit_ratio, final_ndcg = evaluate.metrics(y_model, test_loader, config.getint('EVALUATION', 'top_k'))
    print('-'*20)
    print('>> [Final Test on K] MAE = {:.4f}, MSE = {:.4f}, RMSE = {:.4f}, R2 = {:.4f}'.format(final_mae, final_mse, final_rmse, final_r2))
    print('>> [Final Test on Y] HR = {:.4f}, NDCG = {:.4f}'.format(final_hit_ratio, final_ndcg))

    result_dict = {
        'k_test_result': {'test_mae':float(final_mae),'test_mse':float(final_mse),'test_rmse':float(final_rmse),'test_r2':float(final_r2)},
        'y_test_result': {'test_hr':final_hit_ratio,'test_ndcg':final_ndcg},
        'best_kepoch':best_kepoch,
        'best_yepoch':best_yepoch,
        'ktest_epoch_data':kepoch_data,
        'ytest_epoch_data':yepoch_data
    }

    return result_dict, [best_kmodel_state_dict,best_ymodel_state_dict]


def main():
    model = "KDMF"
    config = configparser.ConfigParser()
    config.read('conf/config.{}.ini'.format(model.lower()))
    print('> [main] THE CONFIG: \n',{section: dict(config[section]) for section in config.sections()})
    print('='*30)

    variant = config['EVALUATION']['variant']
    config_list, variant_list = get_comparison_configlist(config,variant)
    print('> [main] Variant - ',variant,':',variant_list)
    print('='*30)

    for config in config_list:
        data_splitter = data.DataSplitter(config)
        print('> [main] FINISHED data loading(): overall user number: {}, overall item number: {}, overall knowledge number: {}'.format(data_splitter.n_user, data_splitter.n_item, data_splitter.n_knowledge))
        
        validation_loader = data_splitter.validation_loader
        test_loader = data_splitter.test_loader
        train_loader = data_splitter.train_loader
        k_test_loader = data_splitter.k_test_loader
        k_train_loader = data_splitter.k_train_loader

        batch_size = config.getint('MODEL', 'batch_size')
        kbatch_size = config.getint('MODEL', 'kbatch_size')
        l2_reg = config.getfloat('MODEL','l2_reg')
        kl2_reg = config.getfloat('MODEL','kl2_reg')
        lr = config.getfloat('MODEL','lr')
        klr = config.getfloat('MODEL','klr')
        dropout = config.getfloat('MODEL','dropout')
        num_layers = config.getint('MODEL','num_layers')
        latent_dim = config.getint('MODEL','latent_dim')
        klatent_dim = config.getint('MODEL','klatent_dim')
        use_priork, use_targetk, use_itemk = config.getboolean('MODEL','use_priork'), config.getboolean('MODEL','use_targetk'), config.getboolean('MODEL','use_itemk')
        
        result_dir = generate_result_path(config, pretrain=False)
        if not os.path.exists(result_dir):
            '''
            START MODEL TRAINING
            '''
            print('> [main]<K model setting> kbatch_size = {}, klr = {}, klatent_dim = {}, kl2_reg = {}'.format(kbatch_size, klr, klatent_dim, kl2_reg))
            k_model = KnowledgeMF(data_splitter.n_knowledge, klatent_dim).to('cuda:0') # n, latent factor
            print("> [TEST] k_model embed shape:",k_model.embed_k_GMF.weight.data.shape)
            k_opt = optim.Adam(k_model.parameters(), lr=klr, weight_decay=kl2_reg)
            k_criterion = nn.MSELoss()

            print('> [main]<Y model setting> batch_size = {}, lr = {}, latent_dim = {}, l2_reg = {}, num_layers = {}, dropout = {}'.format(batch_size, lr, latent_dim, l2_reg,num_layers,dropout))
            y_model = KDMLP(data_splitter.n_user, data_splitter.n_item, data_splitter.n_knowledge, latent_dim, latent_dim, num_layers=num_layers, dropout=dropout, use_priork=use_priork, use_targetk=use_targetk, use_itemk=use_itemk).to('cuda:0')
            print("> [TEST] y_model embed shape:",y_model.embed_k_MLP.weight.data.shape)
            y_opt = optim.Adam(y_model.parameters(), lr=lr, weight_decay=l2_reg)
            y_criterion = nn.BCEWithLogitsLoss()

            result_dict, best_model_state_dict = train(k_model, k_opt, k_criterion, k_train_loader, k_test_loader, y_model, y_opt, y_criterion, train_loader, validation_loader, test_loader, config)
            if config.getboolean('OUTPUT','save_result'):
                save_train_result(best_model_state_dict, result_dict, config)
            '''
            END MODEL TRAINING
            '''
            print("> [main] FINISHED training.")
        else: # load the trained model
            k_model = KnowledgeMF(data_splitter.n_knowledge, klatent_dim).to('cuda:0') # n, latent factor
            k_model.load_state_dict(torch.load(os.path.join(result_dir, 'k_model.pth')))
            print('> [main] Load the k model from', result_dir)
            y_model = KDMLP(data_splitter.n_user, data_splitter.n_item, data_splitter.n_knowledge, latent_dim, klatent_dim, num_layers=num_layers, dropout=dropout, use_priork=use_priork, use_targetk=use_targetk, use_itemk=use_itemk).to('cuda:0')
            y_model.load_state_dict(torch.load(os.path.join(result_dir, 'y_model.pth')))
            print('> [main] Load the y model from', result_dir)

            final_mae, final_mse, final_rmse, final_r2 = evaluate.klinking_evaluate(k_model, k_test_loader)
            final_hit_ratio, final_ndcg = evaluate.metrics(y_model, test_loader, config.getint('EVALUATION', 'top_k'))
            print('-'*20)
            print('>> [Final Test on K] MAE = {:.4f}, MSE = {:.4f}, RMSE = {:.4f}, R2 = {:.4f}'.format(final_mae, final_mse, final_rmse, final_r2))
            print('>> [Final Test on Y] HR = {:.4f}, NDCG = {:.4f}'.format(final_hit_ratio, final_ndcg))

if __name__ == "__main__":
    main()
