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
from utility.train_util import generate_result_path, explain_result_path, save_train_result, get_comparison_configlist
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train_k(model, opt, criterion, train_loader, test_loader, config):
    epoch_data = []
    init_mae, init_mse, init_rmse, init_r2 = evaluate.klinking_evaluate(model, test_loader)
    epoch_data.append({'epoch': -1, 'loss': '', 'MAE': float(init_mae), 'MSE': float(init_mse), 'RMSE': float(init_rmse), 'R2': float(init_r2)})
    best_model_state_dict = None
    best_r2 = -100000
    best_epoch = 0
    for epoch in range(config.getint('MODEL', 'epoch')):
        model.train()
        total_loss = 0
        for batch in train_loader:
            fromks, toks, scores = batch[0], batch[1], batch[2].float() # tensor
            fromks = fromks.to('cuda:0')
            toks = toks.to('cuda:0')
            scores = scores.to('cuda:0')
            
            # setting
            opt.zero_grad()
            pred = model(fromks, toks)
            loss = criterion(pred.view(-1), scores)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        mae, mse, rmse, r2 = evaluate.klinking_evaluate(model, test_loader)
        epoch_data.append({'epoch': epoch, 'loss': total_loss, 'MAE': float(mae), 'MSE': float(mse), 'RMSE': float(rmse), 'R2': float(r2)})
        if r2 > best_r2:
            best_r2 = r2
            best_epoch = epoch
            best_model_state_dict = copy.deepcopy(model.state_dict())
        print('>> [Epoch {}] Loss = {:.2f}, MAE = {:.4f}, MSE = {:.4f}, RMSE = {:.4f}, R2 = {:.4f}'.format(epoch, total_loss, mae, mse, rmse, r2))
    
    print('>> [Training] the best epoch is {}'.format(best_epoch))
    model.load_state_dict(best_model_state_dict)
    final_mae, final_mse, final_rmse, final_r2 = evaluate.klinking_evaluate(model, test_loader)
    print('-'*20)
    print('>> [Final Test] MAE = {:.4f}, MSE = {:.4f}, RMSE = {:.4f}, R2 = {:.4f}'.format(final_mae, final_mse, final_rmse, final_r2))
    result_dict = {
        'test_result': {'test_mae':float(final_mae),'test_mse':float(final_mse),'test_rmse':float(final_rmse),'test_r2':float(final_r2)},
        'best_epoch':best_epoch,
        'test_epoch_data':epoch_data
    }
    return result_dict, best_model_state_dict


def main():
    method = "KnowledgeMF"
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
        data_splitter = data.DataSplitter(config,task='KLP')
        print('> [main] FINISHED data loading(): overall knowledge number: {}'.format(data_splitter.n_knowledge))
        test_loader = data_splitter.k_test_loader
        train_loader = data_splitter.k_train_loader

        kbatch_size = config.getint('MODEL', 'kbatch_size')
        kl2_reg = config.getfloat('MODEL','kl2_reg')
        klr = config.getfloat('MODEL','klr')
        klatent_dim = config.getint('MODEL','klatent_dim')
        
        result_dir = generate_result_path(config)
        if not os.path.exists(result_dir):
            '''
            START MODEL TRAINING
            '''
            print('> [main]<model setting> kbatch_size = {}, klr = {}, klatent_dim = {}, kl2_reg = {}'.format(kbatch_size, klr, klatent_dim, kl2_reg))
            model = KnowledgeMF(data_splitter.n_knowledge, klatent_dim) # n, latent factor
            model.to('cuda:0')

            opt = optim.Adam(model.parameters(), lr=klr, weight_decay=kl2_reg)
            # criterion = nn.BCEWithLogitsLoss()
            criterion = nn.MSELoss()

            result_dict, best_model_state_dict = train_k(model, opt, criterion, train_loader, test_loader, config)
            if config.getboolean('OUTPUT','save_result'):
                save_train_result(best_model_state_dict, result_dict, config)
            '''
            END MODEL TRAINING
            '''
            print("> [main] FINISHED training.")
        else: # load the trained model
            model = KnowledgeMF(data_splitter.n_knowledge, klatent_dim) # n, latent factor
            model.to('cuda:0')
            model.load_state_dict(torch.load(os.path.join(result_dir, 'model.pth')))
            print('> [main] Load the model from', result_dir)
            final_mae, final_mse, final_rmse, final_r2 = evaluate.klinking_evaluate(model, test_loader)
            print('-'*20)
            print('>> [Final Test] MAE = {:.4f}, MSE = {:.4f}, RMSE = {:.4f}, R2 = {:.4f}'.format(final_mae, final_mse, final_rmse, final_r2))

    # best_model, best_params = find_best_model(config, data_splitter.n_user, data_splitter.n_item)


if __name__ == "__main__":
    main()
