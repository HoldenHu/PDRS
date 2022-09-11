import os
import time
import json
import tqdm
import configparser
import numpy as np
import scipy.sparse as sp

from utility import evaluate
from utility import data
from utility.train_util import save_train_result, generate_result_path

from model.ItemKNN import ItemKNN


def main():
    method = "ItemKNN"
    print("> [main] START the prgram.")
    start = time.time()

    config = configparser.ConfigParser()
    config.read('conf/config.{}.ini'.format(method.lower()))
    print('> [main] THE CONFIG: \n',{section: dict(config[section]) for section in config.sections()})
    print('='*30)

    data_splitter = data.DataSplitter(config)
    done = time.time()
    print('> [main] FINISHED data loading({}): overall user number: {}, overall item number: {}'.format(done-start,data_splitter.n_user, data_splitter.n_item))
    start = done
    
    validation_loader = data_splitter.validation_loader
    test_loader = data_splitter.test_loader
    train_mat = data_splitter.load_train_rating_matrix()
    print("> [main] FINISHED data preparation.")

    result_dir = generate_result_path(config)
    if not os.path.exists(result_dir):
        '''
        START MODEL INIT
        '''
        k_neighbors = int(np.sqrt(data_splitter.n_item))
        model = ItemKNN(k_neighbors, train_mat)
        done = time.time()
        print("> [main] FINISHED ItemKNN model initialization.({})".format(done-start))
        start = done

        final_hit_ratio, final_ndcg = evaluate.metrics(model, test_loader, config.getint('EVALUATION', 'top_k'))
        print('>> [Final Test] HR = {:.4f}, NDCG = {:.4f}'.format(final_hit_ratio, final_ndcg))
        result_dict = {
        'test_result': {'test_hr':final_hit_ratio,'test_ndcg':final_ndcg}
        }
        preloaded_variable_dict = {
            'user_hist':model.user_hist,
            'item_similarity_matrix':model.item_similarity_matrix,
            'similar_items':model.similar_items
        }
        if config.getboolean('OUTPUT','save_result'):
            save_train_result(preloaded_variable_dict, result_dict, config)
    
        done = time.time()
        print("> [main] FINISHED Evaluate model.({})".format(done-start))
    else:
        model_variable_path = os.path.join(result_dir, 'model.npy')
        model = ItemKNN(k_neighbors, train_mat, np.load(model_variable_path, allow_pickle=True).item())
        done = time.time()
        print("> [main] FINISHED ItemKNN model initialization.({})".format(done-start))
        start = done

        final_hit_ratio, final_ndcg = evaluate.metrics(model, test_loader, config.getint('EVALUATION', 'top_k'))
        print('>> [Final Test] HR = {:.4f}, NDCG = {:.4f}'.format(final_hit_ratio, final_ndcg))

        done = time.time()
        print("> [main] FINISHED Evaluate model.({})".format(done-start))
   

if __name__ == "__main__":
    main()
