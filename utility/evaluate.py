import numpy as np
import torch
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def klinking_evaluate(model, test_loader):
    MAE, MSE, RMSE, R2 = [], [],[],[]
    for row in test_loader:
        fromk = row[0].cuda()
        tok = row[1].cuda()
        score = row[2].cpu()

        predictions = model.forward(fromk, tok).cpu()
        predictions = predictions.detach().numpy()

        mae = mean_absolute_error(score, predictions)
        mse = mean_squared_error(score, predictions)
        rmse = sqrt(mean_squared_error(score, predictions))
        r2 = r2_score(score, predictions)
        MAE.append(mae)
        MSE.append(mse)
        RMSE.append(rmse)
        R2.append(r2)
    return np.mean(MAE), np.mean(MSE), np.mean(RMSE), np.mean(R2)

def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k):
	HR, NDCG = [], []

	# for user, item, label, prior, target in test_loader:
	for row in test_loader:
		user = row[0]
		item = row[1]
		label = row[2]

		user = user.cuda()
		item = item.cuda()

		if model.get_alias() in ["ItemPop","ItemKNN","MLP","MF","NCF"]:
			predictions = model.forward(user, item)	
		elif model.get_alias() == "KDMLP":
			user_priork, user_targetk, item_containk = row[3], row[4], row[5] # tensor
			valid_priork_num, valid_targetk_num, valid_itemk_num = row[6], row[7], row[8] # list
			predictions = model.forward(user, item, user_priork, user_targetk, item_containk, valid_priork_num, valid_targetk_num, valid_itemk_num)
			
		else: # ItemPop. ItemKNN
			predictions = model.forward(user, item)	
		_, indices = torch.topk(predictions, top_k)
		indices = indices.cuda()
		recommends = torch.take(item, indices).cpu().numpy().tolist()

		gt_item = item[0].item()
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))

	return np.mean(HR), np.mean(NDCG)