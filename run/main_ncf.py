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
from model.NCF import NCF
from utility.train_util import generate_result_path, explain_result_path, save_train_result, get_comparison_configlist
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train_y(model, opt, criterion, train_loader, validation_loader, test_loader, config):
    epoch_data = []
    init_hit_ratio, init_ndcg = evaluate.metrics(model, validation_loader, config.getint('EVALUATION', 'top_k'))
    epoch_data.append({'epoch': -1, 'loss': '', 'HR': init_hit_ratio, 'NDCG': init_ndcg})
    best_model_state_dict = None
    best_ndcg = 0
    best_epoch = 0
    for epoch in range(config.getint('MODEL', 'epoch')):
        model.train()
        total_loss = 0
        for batch in train_loader:
            users, items, ratings = batch[0], batch[1], batch[2].float() # tensor
            users = users.to('cuda:0')
            items = items.to('cuda:0')
            ratings = ratings.to('cuda:0')
            
            # setting
            opt.zero_grad()
            pred = model(users, items)
            loss = criterion(pred.view(-1), ratings)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        hit_ratio, ndcg = evaluate.metrics(model, validation_loader, config.getint('EVALUATION', 'top_k'))
        epoch_data.append({'epoch': epoch, 'loss': total_loss, 'HR': hit_ratio, 'NDCG': ndcg})
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_epoch = epoch
            best_model_state_dict = copy.deepcopy(model.state_dict())
        print('>> [Epoch {}] Loss = {:.2f}, HR = {:.4f}, NDCG = {:.4f}'.format(epoch, total_loss, hit_ratio, ndcg))
    
    model.load_state_dict(best_model_state_dict)
    final_hit_ratio, final_ndcg = evaluate.metrics(model, test_loader, config.getint('EVALUATION', 'top_k'))
    print('-'*20)
    print('>> [Final Test] HR = {:.4f}, NDCG = {:.4f}'.format(final_hit_ratio, final_ndcg))
    result_dict = {
        'test_result': {'test_hr':final_hit_ratio,'test_ndcg':final_ndcg},
        'best_epoch':best_epoch,
        'validation_epoch_data':epoch_data
    }
    return result_dict, best_model_state_dict



def main():
    method = "NCF"
    pretrain = True # 是否当前为了training
    config = configparser.ConfigParser()
    config.read('conf/config.{}.ini'.format(method.lower()))
    print('> [main] THE CONFIG: \n',{section: dict(config[section]) for section in config.sections()})
    print('='*30)
    
    variant = config['EVALUATION']['variant']
    config_list, variant_list = get_comparison_configlist(config,variant)
    print('> [main] Variant - ',variant,':',variant_list)
    print('='*30)
    for config in config_list:
        '''
        ADJUST PARAMS
        '''
        data_splitter = data.DataSplitter(config)
        print('> [main] FINISHED data loading(): overall user number: {}, overall item number: {}'.format(data_splitter.n_user, data_splitter.n_item))

        n_neg_test = config.getint('DATA', 'n_neg_test') # defaultly, 99
        batch_size = config.getint('MODEL', 'batch_size')
        validation_loader = data_splitter.validation_loader
        test_loader = data_splitter.test_loader
        train_loader = data_splitter.train_loader
        
        l2_reg = config.getfloat('MODEL','l2_reg')
        lr = config.getfloat('MODEL','lr')
        dropout = config.getfloat('MODEL','dropout')
        num_layers = config.getint('MODEL','num_layers')
        latent_dim = config.getint('MODEL','latent_dim')

        
        result_dir = generate_result_path(config)
        if not os.path.exists(result_dir):
            '''
            START MODEL TRAINING
            '''
            print('> [main]<model setting> batch_size = {}, lr = {}, latent_dim = {}, l2_reg = {}, num_layers = {}, dropout = {}'.format(batch_size, lr, latent_dim, l2_reg, num_layers, dropout))
            model = NCF(data_splitter.n_user, data_splitter.n_item, latent_dim, num_layers=num_layers, dropout=dropout)
            model.to('cuda:0')

            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
            criterion = nn.BCEWithLogitsLoss()

            result_dict, best_model_state_dict = train_y(model, opt, criterion, train_loader, validation_loader, test_loader, config)
            if config.getboolean('OUTPUT','save_result'):
                save_train_result(best_model_state_dict, result_dict, config)
            '''
            END MODEL TRAINING
            '''
            print("> [main] FINISHED training.")
        else: # load the trained model
            model = NCF(data_splitter.n_user, data_splitter.n_item, latent_dim, num_layers=num_layers, dropout=dropout)
            model.to('cuda:0')
            model.load_state_dict(torch.load(os.path.join(result_dir, 'model.pth')))
            print('> [main] Load the model from', result_dir)
            final_hit_ratio, final_ndcg = evaluate.metrics(model, test_loader, config.getint('EVALUATION', 'top_k'))
            print('-'*20)
            print('>> [Final Test] HR = {:.4f}, NDCG = {:.4f}'.format(final_hit_ratio, final_ndcg))


if __name__ == "__main__":
    main()
