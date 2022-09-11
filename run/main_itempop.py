import os
import time
import json
import tqdm
import configparser
import scipy.sparse as sp

from utility import evaluate
from utility import data

from model.ItemPop import ItemPop
from utility.train_util import save_train_result, generate_result_path

def main():
    method = "ItemPop"
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
    done = time.time()
    print("> [main] FINISHED data preparation.({})".format(done-start))
    start = done

    result_dir = generate_result_path(config)
    if not os.path.exists(result_dir):
        '''
        START MODEL INIT
        '''
        model = ItemPop(train_mat)
        done = time.time()
        print("> [main] FINISHED ItemPop model initialization.({})".format(done-start))
        start = done

        final_hit_ratio, final_ndcg = evaluate.metrics(model, test_loader, config.getint('EVALUATION', 'top_k'))
        print('>> [Final Test] HR = {:.4f}, NDCG = {:.4f}'.format(final_hit_ratio, final_ndcg))
        result_dict = {
        'test_result': {'test_hr':final_hit_ratio,'test_ndcg':final_ndcg}
        }
        if config.getboolean('OUTPUT','save_result'):
            save_train_result(model.item_ratings, result_dict, config)

        done = time.time()
        print("> [main] FINISHED Evaluate model.({})".format(done-start))
    else:
        model = ItemPop(train_mat, np.load(os.path.join(result_dir, 'model.npy')))
        done = time.time()
        print("> [main] FINISHED ItemPop model initialization.({})".format(done-start))
        start = done

        final_hit_ratio, final_ndcg = evaluate.metrics(model, test_loader, config.getint('EVALUATION', 'top_k'))
        print('>> [Final Test] HR = {:.4f}, NDCG = {:.4f}'.format(final_hit_ratio, final_ndcg))

        done = time.time()
        print("> [main] FINISHED Evaluate model.({})".format(done-start))


if __name__ == "__main__":
    main()
