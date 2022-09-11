import matplotlib.pyplot as plt
import json
import configparser
import os

def plot_compare_statistics(model_out_list,y_label,fig_alias,title ='SSG Data',x_label='epoch'):
    linestyle_list = ['-','-.','--','-','-']
    marker_list = [None,None,'o','v','s']
    fig, ax = plt.subplots()

    for i in range(len(model_out_list)):
        each_model_out = model_out_list[i]
        label = each_model_out['label']
        step_list = each_model_out['step_list']
        value_list = each_model_out['value_list']
        ax.plot(step_list, value_list, linestyle=linestyle_list[i], marker = marker_list[i], label=label)

    ax.set(xlabel=x_label, ylabel=y_label,
        title=title)
    
    ax.grid()
    # 设置刻度字体大小
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15,loc='lower right')
    # plt.show()
    path = "out/figs/"
    plt.savefig(path+fig_alias+".png")
    return

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    neg_strategy = ['random','cluster']
    
    model_out_list = []
    latent_dim = 64
    lr = 0.001
    batch_size = 128
    # for latent_dim in map(int, config['MODEL']['latent_dim'].split()):
    for i in range(1):
        result_dir = "out/train_result/negstrategy_{}-splitstrategy_{}-batch_size_{}-lr_{}-latent_dim_{}-l2_reg_{}-epoch_{}-n_negative_{}-top_k_{}".format(
                                config['DATA']['negative_strategy'], config['DATA']['split_strategy'], batch_size, lr, latent_dim, config['MODEL']['l2_reg'], config['MODEL']['epoch'], config['DATA']['n_neg_train'], config['EVALUATION']['top_k'])

        output_epchos_dict = json.load(open(os.path.join(result_dir, 'epoch_data.json')))
        label = 'latent dimension:'+str(latent_dim)
        step_list = []
        value_list = []
        for each_dict in output_epchos_dict:
            step_list.append(each_dict['epoch'])
            value_list.append(each_dict['NDCG'])

        model_out_list.append({'label':label,'step_list':step_list, 'value_list':value_list})
    
    plot_compare_statistics(model_out_list,'NDCG@10','latent-ndcg',title ='Latent Dimension Comparison')

# "epoch": 0,
# "loss": 224.20529612898827,
# "HR": 0.6202766291173535,
# "NDCG": 0.5017987700854213

if __name__ == "__main__":
    main()