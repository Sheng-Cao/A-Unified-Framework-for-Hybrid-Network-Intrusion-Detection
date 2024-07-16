from copy import copy
import random
import time
import numpy as np
import torch
import os
import argparse
import xgboost as xgb
from torch import cdist, nn
from dataloader import DATA_LOADER
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score, confusion_matrix, \
    classification_report
from tqdm import trange
from models import Embedding_Net

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# required functions
def map_label(label, classes):
    mapped_label = torch.zeros_like(label, dtype=torch.long)
    for i, class_label in enumerate(classes):
        mapped_label[label == class_label] = i
    return mapped_label.to(device)


def compute_per_class_acc(test_label, predicted_label, nclass):
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    for i in range(nclass):
        idx = (test_label == i)
        acc_per_class[i] = float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
        print("predicted= ", float(torch.sum(test_label[idx] == predicted_label[idx])), "real= ", float(torch.sum(idx)))
    print(acc_per_class)

    return acc_per_class.mean()


def indicator(K_matrix):
    K_matrix = K_matrix.long()

    # 获取权重值
    weights_in = weights[K_matrix]

    # 对20个近邻的权重求和
    weighted_unseen_count = weights_in.sum(dim=1)

    return weighted_unseen_count


# def indicator(K_matrix, sentry):
#     seen_count = (K_matrix >= sentry).sum(dim=1)
#     unseen_count = K_matrix.shape[1] - seen_count
#     # score = torch.where(seen_count < weight_test * unseen_count, torch.tensor(0.0, device=K_matrix.device), torch.tensor(1.0, device=K_matrix.device))
#     score = unseen_count
#     return score

def evaluation(y_true, score, option):
    if option == 3:
        report = classification_report(y_true, score)
        print(report)
        return
    # calculate AUC-ROC
    auc_roc = roc_auc_score(y_true, score)
    print(f"AUC-ROC: {auc_roc}")

    # calculate Precision-Recall and AUC-PR
    precision, recall, _ = precision_recall_curve(y_true, score)
    auc_pr = auc(recall, precision)
    print(f"AUC-PR: {auc_pr}")

    # choose a threshhold
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_f1_index = np.argmax(f1_scores)
    best_threshold = _[best_f1_index]
    print(best_threshold)
    y_pred = (score >= best_threshold).astype(int)

    # calculate F1-Score
    f1 = f1_score(y_true, y_pred)
    print(f"F1-Score: {f1}")

    # calculate Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")

    # calculate Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Visulization of Precision-Recall Curve
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

    # Calculate Class-wise Recall
    recall_per_class = {}
    y_pred = torch.from_numpy(y_pred).to(device)

    if option == 0:  # detector
        for cls in dataset.allclasses:
            if cls.item() == 0:
                tp = torch.sum((y_pred == 0) & (dataset.test_label == 0)).item()
                fn = torch.sum((y_pred == 1) & (dataset.test_label == 0)).item()
            else:
                tp = torch.sum((y_pred == 1) & (dataset.test_label == cls)).item()
                fn = torch.sum((y_pred == 0) & (dataset.test_label == cls)).item()

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recall_per_class[cls.item()] = recall

    elif option == 1:  # discrimination
        for cls in dataset.maliciousclasses:
            if cls in dataset.seenclasses:
                tp = torch.sum((y_pred == 0) & (dataset.test_label[dataset.benign_size_test:] == cls)).item()
                fn = torch.sum((y_pred == 1) & (dataset.test_label[dataset.benign_size_test:] == cls)).item()
            else:
                tp = torch.sum((y_pred == 1) & (dataset.test_label[dataset.benign_size_test:] == cls)).item()
                fn = torch.sum((y_pred == 0) & (dataset.test_label[dataset.benign_size_test:] == cls)).item()

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recall_per_class[cls.item()] = recall

    print("Recall per class:")
    for cls, recall in recall_per_class.items():
        print(f"Class {dataset.attacks[cls]}: Recall = {recall}")
    return y_pred


parser = argparse.ArgumentParser()

# set hyperparameters
parser.add_argument('--dataset', default='9')
parser.add_argument('--matdataset', default=True)
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--manualSeed', type=int, default=42)
parser.add_argument('--resSize', type=int, default=70, help='size of visual features')
parser.add_argument('--embedSize', type=int, default=128, help='size of embedding h')
parser.add_argument('--outzSize', type=int, default=32, help='size of non-liner projection z')

opt = parser.parse_args()

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set seed
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# loading data
dataset = DATA_LOADER(opt)

# 1st step: Detect malicious traffic
from adbench.baseline.Supervised import supervised

model = supervised(seed=42, model_name='CatB')  # initialization

# training proceduce of detector
print("start fittting detector")
start_time = time.time()
model.fit(dataset.train_feature.cpu().numpy(), dataset.binary_train.cpu().numpy())  # fit
end_time = time.time()
print("end fittting detector")
print("Training time of the detector：%.4f seconds" % (end_time - start_time))

# inferece proceduce of detector
print("start evaluating detector")
start_time = time.time()
detector_score = model.predict_score(dataset.test_feature.cpu().numpy())  # predict
end_time = time.time()
# evaluation of detector
y_true = dataset.binary_test.cpu().numpy()
detector_prediction = evaluation(y_true, detector_score, 0)
print("end evaluating detector")
print("Inference time of the detector：%.4f seconds" % (end_time - start_time))

# 2nd step: Discriminate unknown categories traffic
test_feature = dataset.all_malicious_feature[dataset.train_seen_feature.shape[0]:]

# sentry denotes whether it belongs to the test set or training set
sentry = test_feature.shape[0]

# initailize hyperparameters and matrixes
pairwise = nn.PairwiseDistance(p=2)
K = 20
indice_matrix = torch.IntTensor(size=(test_feature.shape[0], K)).to(device)

# 获取每个标签及其对应的数量
unique_labels, counts = torch.unique(dataset.train_seen_label, return_counts=True)

# 初始化权值矩阵
train_weights = torch.zeros_like(dataset.train_seen_label, dtype=torch.float32).to(device)

# 遍历每个标签及其对应的数量
# for label, count in zip(unique_labels, counts):
#     if count.item() < 100:
#         train_weights[dataset.train_seen_label == label] = -K/count

test_weights = torch.ones_like(dataset.test_seen_unseen_label, dtype=torch.float32).to(device)
weights = torch.cat((train_weights, test_weights), dim=0)

# training proceduce of discriminator
print("start fittting and evaluating discriminator")
start_time = time.time()
for i in trange(test_feature.shape[0]):
    expand_feature = test_feature[i].unsqueeze(0).expand_as(dataset.all_malicious_feature)
    dis = pairwise(expand_feature, dataset.all_malicious_feature)
    # sort and selection
    _, indices = torch.topk(dis, k=K, largest=False)
    indice_matrix[i] = indices

# inference proceduce of discriminator
discriminator_score = indicator(indice_matrix)

end_time = time.time()

print("Training and inference time of the discriminator：%.4f seconds" % (end_time - start_time))

y_true = dataset.test_seen_unseen_label.cpu().numpy()

discriminator_score = discriminator_score.cpu().numpy()

discriminator_prediction = evaluation(y_true, discriminator_score, 1)
print("end fittting and evaluating discriminator")

# 3rd step: Classify known categories traffic

# training proceduce of known_class_classifier
known_class_classifier = xgb.XGBClassifier()
known_class_classifier.fit(dataset.train_seen_feature.cpu().numpy(),
                           map_label(dataset.train_seen_label, dataset.seenclasses).cpu().numpy())
# inferece proceduce of known_class_classifier
known_preds = known_class_classifier.predict(dataset.test_seen_feature.cpu().numpy())
# evaluation of known_class_classifier
evaluation(map_label(dataset.test_seen_label, dataset.seenclasses).cpu().numpy(), known_preds, 3)

# load unknown_class_classifier
unknown_class_classifier = xgb.XGBClassifier()
# booster = xgb.Booster()
# unknown_class_classifier._Booster = booster
unknown_class_classifier.load_model('./models/' + opt.dataset + '/cls.model')
mapper = Embedding_Net(opt).to(device)
mapper.load_state_dict(torch.load('./models/' + opt.dataset + '/map.pt'))
mapper.eval()
# inferece proceduce of unknown_class_classifier
with torch.no_grad():
    embed, _ = mapper(dataset.test_unseen_feature)

unknown_preds = unknown_class_classifier.predict(embed.cpu().numpy())
# evaluation of unknown_class_classifier
evaluation(map_label(dataset.test_unseen_label, dataset.novelclasses).cpu().numpy(), unknown_preds, 3)

# score_for_malicious = detector_prediction[dataset.benign_size_test:]
# score_for_seen = discriminator_prediction[:dataset.test_seen_feature.shape[0]]
# score_for_unseen = discriminator_prediction[dataset.test_seen_feature.shape[0]:]

# mask_all = (score_for_malicious == 0).cuda()
# mask_seen = (score_for_seen == 1).cuda()
# mask_all_tail = mask_all[:dataset.test_seen_feature.shape[0]]

# # 将mask_all_tail与mask_seen求并集
# final_mask = mask_all_tail | mask_seen
# preds[final_mask.cpu().numpy()] = 100
# evaluation(map_label(dataset.test_seen_label, dataset.seenclasses).cpu().numpy(), preds, 3)


# mask_unseen = (score_for_unseen == 1 or score_for_malicious == 0).cuda()
# test_label = copy.deepcopy(dataset.test_unseen_feature)
# test_label[mask_unseen] = 100
# mask_seen = (score_for_seen == 0 or score_for_malicious == 0).cuda()
# preds[mask_seen] = 100
# evaluation(map_label(dataset.test_seen_label, dataset.seenclasses).cpu().numpy(), preds, 3)

# target_classes = dataset.seenclasses.cuda()
# acc_seen = compute_per_class_acc(util.map_label(dataset.test_seen_label, target_classes), preds, target_classes.size(0))
# print("seen_acc = ", acc_seen)
# print('\n')
# target_classes = dataset.novelclasses.cuda()
# acc_unseen = compute_per_class_acc(util.map_label(dataset.test_unseen_label, target_classes), util.map_label(test_label, target_classes), target_classes.size(0))
# print("unseen_acc = ",acc_unseen)




