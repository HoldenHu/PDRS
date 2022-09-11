# Author: Hengchang Hu
# Date: 22 April, 2021
# Description: for help the training for RS,KDRS,KLP models
import os
import torch
import json
import copy
import numpy as np

def generate_result_path(config, pretrain=False):
    #  the path should be : model_saving_path + data_type + task + method + [negstrategy|clusternum_splitstratygy_negtrain_negtest_knegtest_'bz'bz_'kbz'kbz_'lr'lr_'l2reg'l2reg_'lf'latentfactor_'drop'dropout_'layer'numlayer_'topk'topk]
    #  example: out/course/KDRS/KDMLP/random10_warm-start_4_99_99_bz128_kbz64_lr0.001_l2reg0_lf64_drop0.0_layer4_topk10
    model_saving_path = config.get('OUTPUT', 'model_saving_path')
    data_type = config.get('DATA', 'data_type')
    method = config.get('MODEL', 'method')
    if 'Knowledge' in method:
        task = 'KLP'
    elif 'KD' in method:
        task = 'KDRS'
    else:
        task = 'RS'
    output_folder = os.path.join(model_saving_path,data_type,task)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_folder = os.path.join(output_folder,method)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if method == 'KDMLP':
        result_folder = '{negative_strategy}_{cluster_num}_{split_strategy}_{n_neg_train}_{n_neg_test}_{n_kneg_test}_bz{batch_size}_kbz{kbatch_size}_lr{lr}_klr{klr}_l2reg{l2_reg}_kl2reg{kl2_reg}_lf{latent_dim}_klf{klatent_dim}_layer{num_layers}_klayer{knum_layers}_drop{dropout}_kdrop{kdropout}_p{use_priork}_t{use_targetk}_i{use_itemk}_topk{top_k}_PRETRAIN{pretrain}'.format(
                        negative_strategy = config['DATA']['negative_strategy'],
                        cluster_num = config['DATA']['cluster_num'],
                        split_strategy = config['DATA']['split_strategy'],
                        n_neg_train = config['DATA']['n_neg_train'],
                        n_neg_test = config['DATA']['n_neg_test'],
                        n_kneg_test = config['DATA']['n_kneg_test'],
                        batch_size = config['MODEL']['batch_size'],
                        kbatch_size = config['MODEL']['kbatch_size'],
                        lr = config['MODEL']['lr'],
                        klr = config['MODEL']['klr'],
                        l2_reg = config['MODEL']['l2_reg'],
                        kl2_reg = config['MODEL']['kl2_reg'],
                        latent_dim = config['MODEL']['latent_dim'],
                        klatent_dim = config['MODEL']['klatent_dim'],
                        num_layers = config['MODEL']['num_layers'],
                        knum_layers = config['MODEL']['knum_layers'],
                        dropout = config['MODEL']['dropout'],
                        kdropout = config['MODEL']['kdropout'],
                        use_priork = config['MODEL']['use_priork'],
                        use_targetk = config['MODEL']['use_targetk'],
                        use_itemk = config['MODEL']['use_itemk'],
                        top_k = config['EVALUATION']['top_k'],
                        pretrain = pretrain
            )
    elif method == 'KDMF':
        result_folder = '{negative_strategy}_{cluster_num}_{split_strategy}_{n_neg_train}_{n_neg_test}_{n_kneg_test}_bz{batch_size}_kbz{kbatch_size}_lr{lr}_klr{klr}_l2reg{l2_reg}_kl2reg{kl2_reg}_lf{latent_dim}_klf{klatent_dim}_layer{num_layers}_drop{dropout}_p{use_priork}_t{use_targetk}_i{use_itemk}_topk{top_k}_PRETRAIN{pretrain}'.format(
                        negative_strategy = config['DATA']['negative_strategy'],
                        cluster_num = config['DATA']['cluster_num'],
                        split_strategy = config['DATA']['split_strategy'],
                        n_neg_train = config['DATA']['n_neg_train'],
                        n_neg_test = config['DATA']['n_neg_test'],
                        n_kneg_test = config['DATA']['n_kneg_test'],
                        batch_size = config['MODEL']['batch_size'],
                        kbatch_size = config['MODEL']['kbatch_size'],
                        lr = config['MODEL']['lr'],
                        klr = config['MODEL']['klr'],
                        l2_reg = config['MODEL']['l2_reg'],
                        kl2_reg = config['MODEL']['kl2_reg'],
                        latent_dim = config['MODEL']['latent_dim'],
                        klatent_dim = config['MODEL']['latent_dim'],
                        num_layers = config['MODEL']['num_layers'],
                        dropout = config['MODEL']['dropout'],
                        use_priork = config['MODEL']['use_priork'],
                        use_targetk = config['MODEL']['use_targetk'],
                        use_itemk = config['MODEL']['use_itemk'],
                        top_k = config['EVALUATION']['top_k'],
                        pretrain = pretrain
            )
    ############################## RS
    elif method == 'ItemPop':
        result_folder = '{negative_strategy}_{cluster_num}_{split_strategy}_{n_neg_train}_{n_neg_test}_topk{top_k}'.format(
                        negative_strategy = config['DATA']['negative_strategy'],
                        cluster_num = config['DATA']['cluster_num'],
                        split_strategy = config['DATA']['split_strategy'],
                        n_neg_train = config['DATA']['n_neg_train'],
                        n_neg_test = config['DATA']['n_neg_test'],
                        top_k = config['EVALUATION']['top_k']
            )
    elif method == 'ItemKNN':
        result_folder = '{negative_strategy}_{cluster_num}_{split_strategy}_{n_neg_train}_{n_neg_test}_topk{top_k}'.format(
                        negative_strategy = config['DATA']['negative_strategy'],
                        cluster_num = config['DATA']['cluster_num'],
                        split_strategy = config['DATA']['split_strategy'],
                        n_neg_train = config['DATA']['n_neg_train'],
                        n_neg_test = config['DATA']['n_neg_test'],
                        top_k = config['EVALUATION']['top_k']
            )
    elif method == 'MLP':
        result_folder = '{negative_strategy}_{cluster_num}_{split_strategy}_{n_neg_train}_{n_neg_test}_{n_kneg_test}_bz{batch_size}_kbz{kbatch_size}_lr{lr}_l2reg{l2_reg}_lf{latent_dim}_layer{num_layers}_drop{dropout}_topk{top_k}'.format(
                        negative_strategy = config['DATA']['negative_strategy'],
                        cluster_num = config['DATA']['cluster_num'],
                        split_strategy = config['DATA']['split_strategy'],
                        n_neg_train = config['DATA']['n_neg_train'],
                        n_neg_test = config['DATA']['n_neg_test'],
                        n_kneg_test = config['DATA']['n_kneg_test'],
                        batch_size = config['MODEL']['batch_size'],
                        kbatch_size = config['MODEL']['kbatch_size'],
                        lr = config['MODEL']['lr'],
                        l2_reg = config['MODEL']['l2_reg'],
                        latent_dim = config['MODEL']['latent_dim'],
                        num_layers = config['MODEL']['num_layers'],
                        dropout = config['MODEL']['dropout'],
                        top_k = config['EVALUATION']['top_k']
            )
    elif method in ['MF','BPR']:
        result_folder = '{negative_strategy}_{cluster_num}_{split_strategy}_{n_neg_train}_{n_neg_test}_{n_kneg_test}_bz{batch_size}_kbz{kbatch_size}_lr{lr}_l2reg{l2_reg}_lf{latent_dim}_topk{top_k}'.format(
                        negative_strategy = config['DATA']['negative_strategy'],
                        cluster_num = config['DATA']['cluster_num'],
                        split_strategy = config['DATA']['split_strategy'],
                        n_neg_train = config['DATA']['n_neg_train'],
                        n_neg_test = config['DATA']['n_neg_test'],
                        n_kneg_test = config['DATA']['n_kneg_test'],
                        batch_size = config['MODEL']['batch_size'],
                        kbatch_size = config['MODEL']['kbatch_size'],
                        lr = config['MODEL']['lr'],
                        l2_reg = config['MODEL']['l2_reg'],
                        latent_dim = config['MODEL']['latent_dim'],
                        top_k = config['EVALUATION']['top_k']
            )
    elif method == 'NCF':
        result_folder = '{negative_strategy}_{cluster_num}_{split_strategy}_{n_neg_train}_{n_neg_test}_{n_kneg_test}_bz{batch_size}_kbz{kbatch_size}_lr{lr}_l2reg{l2_reg}_lf{latent_dim}_layer{num_layers}_drop{dropout}_topk{top_k}'.format(
                        negative_strategy = config['DATA']['negative_strategy'],
                        cluster_num = config['DATA']['cluster_num'],
                        split_strategy = config['DATA']['split_strategy'],
                        n_neg_train = config['DATA']['n_neg_train'],
                        n_neg_test = config['DATA']['n_neg_test'],
                        n_kneg_test = config['DATA']['n_kneg_test'],
                        batch_size = config['MODEL']['batch_size'],
                        kbatch_size = config['MODEL']['kbatch_size'],
                        lr = config['MODEL']['lr'],
                        l2_reg = config['MODEL']['l2_reg'],
                        latent_dim = config['MODEL']['latent_dim'],
                        num_layers = config['MODEL']['num_layers'],
                        dropout = config['MODEL']['dropout'],
                        top_k = config['EVALUATION']['top_k']
            )
    ############################## KLP
    elif method == 'KnowledgeMF':
        result_folder = '{negative_strategy}_{cluster_num}_{split_strategy}_{n_neg_train}_{n_neg_test}_{n_kneg_test}_bz{batch_size}_kbz{kbatch_size}_klr{klr}_kl2reg{kl2_reg}_klf{klatent_dim}_topk{top_k}'.format(
                        negative_strategy = config['DATA']['negative_strategy'],
                        cluster_num = config['DATA']['cluster_num'],
                        split_strategy = config['DATA']['split_strategy'],
                        n_neg_train = config['DATA']['n_neg_train'],
                        n_neg_test = config['DATA']['n_neg_test'],
                        n_kneg_test = config['DATA']['n_kneg_test'],
                        batch_size = config['MODEL']['batch_size'],
                        kbatch_size = config['MODEL']['kbatch_size'],
                        klr = config['MODEL']['klr'],
                        kl2_reg = config['MODEL']['kl2_reg'],
                        klatent_dim = config['MODEL']['klatent_dim'],
                        top_k = config['EVALUATION']['top_k']
            )
    elif method == 'KnowledgeMLP':
        result_folder = '{negative_strategy}_{cluster_num}_{split_strategy}_{n_neg_train}_{n_neg_test}_{n_kneg_test}_bz{batch_size}_kbz{kbatch_size}_klr{klr}_kl2reg{kl2_reg}_klf{klatent_dim}_klayer{knum_layers}_kdrop{kdropout}_topk{top_k}'.format(
                        negative_strategy = config['DATA']['negative_strategy'],
                        cluster_num = config['DATA']['cluster_num'],
                        split_strategy = config['DATA']['split_strategy'],
                        n_neg_train = config['DATA']['n_neg_train'],
                        n_neg_test = config['DATA']['n_neg_test'],
                        n_kneg_test = config['DATA']['n_kneg_test'],
                        batch_size = config['MODEL']['batch_size'],
                        kbatch_size = config['MODEL']['kbatch_size'],
                        klr = config['MODEL']['klr'],
                        kl2_reg = config['MODEL']['kl2_reg'],
                        klatent_dim = config['MODEL']['klatent_dim'],
                        knum_layers = config['MODEL']['knum_layers'],
                        kdropout = config['MODEL']['kdropout'],
                        top_k = config['EVALUATION']['top_k']
            )

    return os.path.join(output_folder,result_folder)

def explain_result_path(result_folder):
    #  example: out/course/KDRS/KDMLP/random10_warm-start_4_99_99_bz128_kbz64_lr0.001_l2reg0_lf64_drop0.0_layer4_topk10
    path = result_folder.split('/')
    data_type = path[1] # course
    task = path[2] # KDRS
    method = path[3] # KDMLP
    result_folder = path[4]
    result = result_folder.split('_')

    return_dict = {}
    if method == 'KDMLP':
        # '{negative_strategy}_{cluster_num}_{split_strategy}_{n_neg_train}_{n_neg_test}_{n_kneg_test}_bz{batch_size}_kbz{kbatch_size}_lr{lr}_klr{klr}_l2reg{l2_reg}_
        # kl2reg{kl2_reg}_lf{latent_dim}_klf{klatent_dim}_layer{num_layers}_klayer{knum_layers}_drop{dropout}_kdrop{kdropout}__p{use_priork}_t{use_targetk}_i{use_itemk}_topk{top_k}_PRETRAIN{pretrain}'
        return_dict['negative_strategy'] = result[0]
        return_dict['cluster_num'] = result[1]
        return_dict['split_strategy'] = result[2]
        return_dict['n_neg_train'] = result[3]
        return_dict['n_neg_test'] = result[4]
        return_dict['n_kneg_test'] = result[5]
        return_dict['batch_size'] = result[6][2:]
        return_dict['kbatch_size'] = result[7][3:]
        return_dict['lr'] = result[8][2:]
        return_dict['klr'] = result[9][3:]
        return_dict['l2_reg'] = result[10][5:]
        return_dict['kl2_reg'] = result[11][6:]
        return_dict['latent_dim'] = result[12][2:]
        return_dict['klatent_dim'] = result[13][3:]
        return_dict['num_layers'] = result[14][5:]
        return_dict['knum_layers'] = result[15][6:]
        return_dict['dropout'] = result[16][4:]
        return_dict['kdropout'] = result[17][5:]
        return_dict['use_priork'] = result[18][1:]
        return_dict['use_targetk'] = result[19][1:]
        return_dict['use_itemk'] = result[20][1:]
        return_dict['top_k'] = result[21][4:]
        return_dict['pretrain'] = result[22][8:]
    elif method == 'KDMF':
        # '{negative_strategy}_{cluster_num}_{split_strategy}_{n_neg_train}_{n_neg_test}_{n_kneg_test}_bz{batch_size}_kbz{kbatch_size}_lr{lr}_klr{klr}_l2reg{l2_reg}_
        # kl2reg{kl2_reg}_lf{latent_dim}_klf{klatent_dim}_layer{num_layers}_drop{dropout}_p{use_priork}_t{use_targetk}_i{use_itemk}_topk{top_k}_PRETRAIN{pretrain}'
        return_dict['negative_strategy'] = result[0]
        return_dict['cluster_num'] = result[1]
        return_dict['split_strategy'] = result[2]
        return_dict['n_neg_train'] = result[3]
        return_dict['n_neg_test'] = result[4]
        return_dict['n_kneg_test'] = result[5]
        return_dict['batch_size'] = result[6][2:]
        return_dict['kbatch_size'] = result[7][3:]
        return_dict['lr'] = result[8][2:]
        return_dict['klr'] = result[9][3:]
        return_dict['l2_reg'] = result[10][5:]
        return_dict['kl2_reg'] = result[11][6:]
        return_dict['latent_dim'] = result[12][2:]
        return_dict['klatent_dim'] = result[13][3:]
        return_dict['num_layers'] = result[14][5:]
        return_dict['dropout'] = result[15][4:]
        return_dict['use_priork'] = result[16][1:]
        return_dict['use_targetk'] = result[17][1:]
        return_dict['use_itemk'] = result[18][1:]
        return_dict['top_k'] = result[19][4:]
        return_dict['pretrain'] = result[20][8:]
    ############################## RS
    elif method == 'ItemPop':
        # '{negative_strategy}_{cluster_num}_{split_strategy}_{n_neg_train}_{n_neg_test}_topk{top_k}'
        return_dict['negative_strategy'] = result[0]
        return_dict['cluster_num'] = result[1]
        return_dict['split_strategy'] = result[2]
        return_dict['n_neg_train'] = result[3]
        return_dict['n_neg_test'] = result[4]
        return_dict['top_k'] = result[5][4:]
    elif method == 'ItemKNN':
        # '{negative_strategy}_{cluster_num}_{split_strategy}_{n_neg_train}_{n_neg_test}_topk{top_k}'
        return_dict['negative_strategy'] = result[0]
        return_dict['cluster_num'] = result[1]
        return_dict['split_strategy'] = result[2]
        return_dict['n_neg_train'] = result[3]
        return_dict['n_neg_test'] = result[4]
        return_dict['top_k'] = result[5][4:]
    elif method == 'MLP':
        # '{negative_strategy}_{cluster_num}_{split_strategy}_{n_neg_train}_{n_neg_test}_{n_kneg_test}_bz{batch_size}_kbz{kbatch_size}_lr{lr}_l2reg{l2_reg}_lf{latent_dim}_layer{num_layers}_drop{dropout}_topk{top_k}'
        return_dict['negative_strategy'] = result[0]
        return_dict['cluster_num'] = result[1]
        return_dict['split_strategy'] = result[2]
        return_dict['n_neg_train'] = result[3]
        return_dict['n_neg_test'] = result[4]
        return_dict['n_kneg_test'] = result[5]
        return_dict['batch_size'] = result[6][2:]
        return_dict['kbatch_size'] = result[7][3:]
        return_dict['lr'] = result[8][2:]
        return_dict['l2_reg'] = result[9][5:]
        return_dict['latent_dim'] = result[10][2:]
        return_dict['num_layers'] = result[11][5:]
        return_dict['dropout'] = result[12][4:]
        return_dict['top_k'] = result[13][4:]
    elif method in ['MF','BPR']:
        # '{negative_strategy}_{cluster_num}_{split_strategy}_{n_neg_train}_{n_neg_test}_{n_kneg_test}_bz{batch_size}_kbz{kbatch_size}_lr{lr}_l2reg{l2_reg}_lf{latent_dim}_topk{top_k}'
        return_dict['negative_strategy'] = result[0]
        return_dict['cluster_num'] = result[1]
        return_dict['split_strategy'] = result[2]
        return_dict['n_neg_train'] = result[3]
        return_dict['n_neg_test'] = result[4]
        return_dict['n_kneg_test'] = result[5]
        return_dict['batch_size'] = result[6][2:]
        return_dict['kbatch_size'] = result[7][3:]
        return_dict['lr'] = result[8][2:]
        return_dict['l2_reg'] = result[9][5:]
        return_dict['latent_dim'] = result[10][2:]
        return_dict['top_k'] = result[11][4:]
    elif method == 'NCF':
        # '{negative_strategy}_{cluster_num}_{split_strategy}_{n_neg_train}_{n_neg_test}_{n_kneg_test}_bz{batch_size}_kbz{kbatch_size}_lr{lr}_l2reg{l2_reg}_lf{latent_dim}_layer{num_layers}_drop{dropout}_topk{top_k}'
        return_dict['negative_strategy'] = result[0]
        return_dict['cluster_num'] = result[1]
        return_dict['split_strategy'] = result[2]
        return_dict['n_neg_train'] = result[3]
        return_dict['n_neg_test'] = result[4]
        return_dict['n_kneg_test'] = result[5]
        return_dict['batch_size'] = result[6][2:]
        return_dict['kbatch_size'] = result[7][3:]
        return_dict['lr'] = result[8][2:]
        return_dict['l2_reg'] = result[9][5:]
        return_dict['latent_dim'] = result[10][2:]
        return_dict['num_layers'] = result[11][5:]
        return_dict['dropout'] = result[12][4:]
        return_dict['top_k'] = result[13][4:]
    elif method == 'KnowledgeMF':
        # '{negative_strategy}_{cluster_num}_{split_strategy}_{n_neg_train}_{n_neg_test}_{n_kneg_test}_bz{batch_size}_kbz{kbatch_size}_klr{klr}_kl2reg{kl2_reg}_klf{klatent_dim}_topk{top_k}'
        return_dict['negative_strategy'] = result[0]
        return_dict['cluster_num'] = result[1]
        return_dict['split_strategy'] = result[2]
        return_dict['n_neg_train'] = result[3]
        return_dict['n_neg_test'] = result[4]
        return_dict['n_kneg_test'] = result[5]
        return_dict['batch_size'] = result[6][2:]
        return_dict['kbatch_size'] = result[7][3:]
        return_dict['klr'] = result[8][3:]
        return_dict['kl2_reg'] = result[9][6:]
        return_dict['klatent_dim'] = result[10][3:]
        return_dict['top_k'] = result[11][4:]
    elif method == 'KnowledgeMLP':
        # '{negative_strategy}_{cluster_num}_{split_strategy}_{n_neg_train}_{n_neg_test}_{n_kneg_test}_bz{batch_size}_kbz{kbatch_size}_klr{klr}_kl2reg{kl2_reg}_klf{klatent_dim}_klayer{knum_layers}_kdrop{kdropout}_topk{top_k}'
        return_dict['negative_strategy'] = result[0]
        return_dict['cluster_num'] = result[1]
        return_dict['split_strategy'] = result[2]
        return_dict['n_neg_train'] = result[3]
        return_dict['n_neg_test'] = result[4]
        return_dict['n_kneg_test'] = result[5]
        return_dict['batch_size'] = result[6][2:]
        return_dict['kbatch_size'] = result[7][3:]
        return_dict['klr'] = result[8][3:]
        return_dict['kl2_reg'] = result[9][6:]
        return_dict['klatent_dim'] = result[10][3:]
        return_dict['knum_layers'] = result[11][6:]
        return_dict['kdropout'] = result[12][5:]
        return_dict['top_k'] = result[13][4:]

    return return_dict


def get_comparison_configlist(config,variant):
    '''
    change one params, while freeze other params
    config: Config
    variant: 'n_neg_train'/'batch_size'/'kbatch_size'/'lr'/'klr'/'latent_dim'/'klatent_dim'/'l2_reg'/'kl2_reg'/'epoch'/'num_layers'/'knum_layers'/'dropout'/'kdropout'/'top_k'
    todo_variant: 'negative_strategy'/'cluster_num'/'split_strategy'
    '''
    configlist = []
    #################################### DATA VARIANT
    if variant == '':
        variant_list = ['None']
        configlist.append(copy.deepcopy(config))
    elif variant == 'negative_strategy': # defaultly random
        variant_list = ['random','cluster-10','cluster-20','cluster-40','cluster-80']
        for negative_strategy in variant_list:
            _ = negative_strategy.split('-')
            if len(_) == 1:
                config['DATA']['negative_strategy'] = _[0]
                config['DATA']['cluster_num'] = '10' # default
                configlist.append(copy.deepcopy(config))
            else:
                config['DATA']['negative_strategy'] = _[0]
                config['DATA']['cluster_num'] = _[1]
                configlist.append(copy.deepcopy(config))
    elif variant == 'n_neg_train': # defaultly 4
        variant_list = ['1','2','3','4','5']
        for n_neg_train in variant_list:
            config['DATA']['n_neg_train'] = n_neg_train
            configlist.append(copy.deepcopy(config))
    #################################### MODEL VARIANT
    elif variant == 'batch_size': # defaultly 128
        # variant_list = ['8','16','32','64','128','256','512']
        variant_list = ['8','32','128','512']
        for batch_size in variant_list:
            config['MODEL']['batch_size'] = batch_size
            configlist.append(copy.deepcopy(config))
    elif variant == 'kbatch_size': # defaultly 64
        variant_list = ['8','32','128','512']
        for kbatch_size in variant_list:
            config['MODEL']['kbatch_size'] = kbatch_size
            configlist.append(copy.deepcopy(config))
    elif variant == 'lr': # defaultly 0.001
        variant_list = ['0.0005','0.001','0.005']
        for lr in variant_list:
            config['MODEL']['lr'] = lr
            configlist.append(copy.deepcopy(config))
    elif variant == 'klr': # defaultly 0.001
        variant_list = ['0.0005','0.001','0.005']
        for klr in variant_list:
            config['MODEL']['klr'] = klr
            configlist.append(copy.deepcopy(config))
    elif variant == 'latent_dim_for_kdmf': # defaultly 64
        num_layers = config.getint('MODEL','num_layers')
        variant_list = ['16','32']
        for latent_dim in variant_list:
            config['MODEL']['latent_dim'] = latent_dim
            klatent_dim = int(latent_dim) * (2 ** (num_layers - 2))
            config['MODEL']['klatent_dim'] = str(klatent_dim)
            configlist.append(copy.deepcopy(config))
    elif variant == 'both_latent_dim':
        variant_list = [('16','16'),('16','64'),('16','256'),('64','16'),('64','64'),('64','256'),('256','16'),('256','64'),('256','256')]
        for lf, klf in variant_list:
            config['MODEL']['latent_dim'] = lf
            config['MODEL']['klatent_dim'] = klf
            configlist.append(copy.deepcopy(config))
    elif variant == 'latent_dim': # defaultly 64
        variant_list = ['16','32','64','128']
        for latent_dim in variant_list:
            config['MODEL']['latent_dim'] = latent_dim
            configlist.append(copy.deepcopy(config))
    elif variant == 'klatent_dim': # defaultly 64
        variant_list = ['16','32','64','128']
        for klatent_dim in variant_list:
            config['MODEL']['klatent_dim'] = klatent_dim
            configlist.append(copy.deepcopy(config))
    elif variant == 'l2_reg': # defaultly 0
        variant_list = ['0','0.0000001']
        for l2_reg in variant_list:
            config['MODEL']['l2_reg'] = l2_reg
            configlist.append(copy.deepcopy(config))
    elif variant == 'kl2_reg': # defaultly 0
        variant_list = ['0','0.0000001']
        for kl2_reg in variant_list:
            config['MODEL']['kl2_reg'] = kl2_reg
            configlist.append(copy.deepcopy(config))
    elif variant == 'epoch': # defaultly 40
        variant_list = ['20','40','80']
        for epoch in variant_list:
            config['MODEL']['epoch'] = epoch
            configlist.append(copy.deepcopy(config))
    elif variant == 'num_layers': # defaultly 4
        variant_list = ['3','4','5','6']
        for num_layers in variant_list:
            config['MODEL']['num_layers'] = num_layers
            configlist.append(copy.deepcopy(config))
    elif variant == 'knum_layers': # defaultly 4
        variant_list = ['3','4','5','6']
        for knum_layers in variant_list:
            config['MODEL']['knum_layers'] = knum_layers
            configlist.append(copy.deepcopy(config))
    elif variant == 'dropout': # defaultly 0.0
        variant_list = ['0.0','0.5'] 
        for dropout in variant_list:
            config['MODEL']['dropout'] = dropout
            configlist.append(copy.deepcopy(config))
    elif variant == 'kdropout': # defaultly 0.0
        variant_list = ['0.0','0.5'] 
        for kdropout in variant_list:
            config['MODEL']['kdropout'] = kdropout
            configlist.append(copy.deepcopy(config))
    
    elif variant == 'use_k': # defaultly True, True, True
        variant_list = [('True','True','True'),('True','True','False'),('True','False','True'),('True','False','False'),('False','True','True'),('False','True','False'),('False','False','True'),('False','False','False')] 
        for usek in variant_list:
            config['MODEL']['use_priork'] = usek[0]
            config['MODEL']['use_targetk'] = usek[1]
            config['MODEL']['use_itemk'] = usek[2]
            configlist.append(copy.deepcopy(config))
    elif variant == 'use_k_0': # defaultly True, True, True
        # variant_list = [('True','True','True'),('True','True','False'),('True','False','True'),('True','False','False'),('False','True','True'),('False','True','False'),('False','False','True'),('False','False','False')] 
        variant_list = [('True','True','True'),('True','True','False')] 
        for usek in variant_list:
            config['MODEL']['use_priork'] = usek[0]
            config['MODEL']['use_targetk'] = usek[1]
            config['MODEL']['use_itemk'] = usek[2]
            configlist.append(copy.deepcopy(config))
    elif variant == 'use_k_1': # defaultly True, True, True
        variant_list = [('True','False','True'),('True','False','False')]
        for usek in variant_list:
            config['MODEL']['use_priork'] = usek[0]
            config['MODEL']['use_targetk'] = usek[1]
            config['MODEL']['use_itemk'] = usek[2]
            configlist.append(copy.deepcopy(config))
    elif variant == 'use_k_2': # defaultly True, True, True
        variant_list = [('False','True','True'),('False','True','False')]
        for usek in variant_list:
            config['MODEL']['use_priork'] = usek[0]
            config['MODEL']['use_targetk'] = usek[1]
            config['MODEL']['use_itemk'] = usek[2]
            configlist.append(copy.deepcopy(config))
    elif variant == 'use_k_3': # defaultly True, True, True
        variant_list = [('False','False','True'),('False','False','False')]
        for usek in variant_list:
            config['MODEL']['use_priork'] = usek[0]
            config['MODEL']['use_targetk'] = usek[1]
            config['MODEL']['use_itemk'] = usek[2]
            configlist.append(copy.deepcopy(config))
    #################################### EVALUATION VARIANT
    elif variant == 'top_k': # defaultly 10
        variant_list = ['1','2','3','4','5','6','7','8','9','10']
        for top_k in variant_list:
            config['EVALUATION']['top_k'] = top_k
            configlist.append(copy.deepcopy(config))
    return configlist, variant_list


def save_train_result(best_model_state_dict, result_dict, config):
    method = config['MODEL']['method'] 
    result_dir = generate_result_path(config)
    os.makedirs(result_dir, exist_ok=True)
    if method == 'ItemPop':
        # the best_model_state_dict is the ItemPop.item_ratings: np.array()
        np.save(os.path.join(result_dir, 'model.npy'), best_model_state_dict)
    elif method == 'ItemKNN':
        # preloaded_variable_dict = {
        #     'user_hist':model.user_hist,
        #     'item_similarity_matrix':model.item_similarity_matrix,
        #     'similar_items':model.similar_items
        # }
        np.save(os.path.join(result_dir, 'model.npy'), best_model_state_dict)
    elif method in ['KDMLP','KDMF']:
        # [best_kmodel_state_dict, best_ymodel_state_dict]
        torch.save(best_model_state_dict[0], os.path.join(result_dir, 'k_model.pth'))
        torch.save(best_model_state_dict[1], os.path.join(result_dir, 'y_model.pth'))
    else:
        torch.save(best_model_state_dict, os.path.join(result_dir, 'model.pth'))

    with open(os.path.join(result_dir, 'result_dict.json'), 'w') as f:
        json.dump(result_dict, f, indent=4)