import random
import time
import numpy as np
import torch
import os
import argparse
import xgboost as xgb
from torch import nn
from dataloader import DATA_LOADER
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score, confusion_matrix, \
    classification_report

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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


def indicator(K_matrix, sentry):
    seen_count = (K_matrix >= sentry).sum(dim=1)
    unseen_count = K_matrix.shape[1] - seen_count
    # score = torch.where(seen_count < weight_test * unseen_count, torch.tensor(0.0, device=K_matrix.device), torch.tensor(1.0, device=K_matrix.device))
    score = unseen_count
    return score


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


parser = argparse.ArgumentParser()

# set hyperparameters
parser.add_argument('--dataset', default='0')
parser.add_argument('--matdataset', default=True)
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--manualSeed', type=int, default=42)

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

model = supervised(seed=42, model_name='XGB')  # initialization

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
score = model.predict_score(dataset.test_feature.cpu().numpy())  # predict
end_time = time.time()
# evaluation of detector
y_true = dataset.binary_test.cpu().numpy()
evaluation(y_true, score, 0)
print("end evaluating detector")
print("Inference time of the detector：%.4f seconds" % (end_time - start_time))

# 2nd step: Discriminate unknown categories traffic
test_feature = dataset.test_feature[dataset.benign_size_test:]
all_feature = torch.cat((test_feature, dataset.train_seen_feature), dim=0).detach()

# sentry denotes whether it belongs to the test set or training set
sentry = test_feature.shape[0]

# initailize hyperparameters and matrixes
pairwise = nn.PairwiseDistance(p=2)
K = 50
indice_matrix = torch.IntTensor(size=(test_feature.shape[0], K)).to(device)
dis_matrix = torch.FloatTensor(size=(test_feature.shape[0], K)).to(device)

# training proceduce of discriminator
print("start fittting and evaluating discriminator")
start_time = time.time()
for i in range(test_feature.shape[0]):
    expand_feature = test_feature[i].unsqueeze(0).expand_as(all_feature)
    dis = pairwise(expand_feature, all_feature)
    # sort and selection
    distances, indices = torch.topk(dis, k=K, largest=False)
    dis_matrix[i] = distances
    indice_matrix[i] = indices

# inference proceduce of discriminator
score = indicator(indice_matrix, sentry)
end_time = time.time()

print("Training and inference time of the discriminator%.4f seconds" % (end_time - start_time))

y_true = dataset.test_seen_unseen_label.cpu().numpy()
score = score.cpu().numpy()
evaluation(y_true, score, 1)
print("end fittting and evaluating discriminator")

# 3rd step: Classify known categories traffic

# training proceduce of classfier
xgbmodel = xgb.XGBClassifier()
xgbmodel.fit(dataset.train_feature.cpu().numpy(), map_label(dataset.train_label, dataset.seenclasses).cpu().numpy())
# inferece proceduce of classfier
preds = xgbmodel.predict(dataset.test_seen_feature.cpu().numpy())
# evaluation of classfier
evaluation(map_label(dataset.test_seen_label, dataset.seenclasses).cpu().numpy(), preds, 3)

# score_for_unseen = score[dataset.test_seen_feature.shape[0]:]
# score_for_seen = score[:dataset.test_seen_feature.shape[0]]
# # training procedure of classifier
# target_classes = dataset.seenclasses.cuda()
# xgbmodel = xgb.XGBClassifier()
# xgbmodel.fit(dataset.train_feature.cpu().numpy(), util.map_label(dataset.train_label, target_classes).cpu().numpy())
# preds = xgbmodel.predict(dataset.test_seen_feature.cpu().numpy())
# preds = torch.from_numpy(preds).cuda()
# acc_seen = compute_per_class_acc(util.map_label(dataset.test_seen_label, target_classes), preds, target_classes.size(0))
# print("seen_acc = ", acc_seen)

# mask_unseen = (score_for_unseen == 1).cuda()
# test_label = copy.deepcopy(dataset.test_unseen_feature)
# test_label[mask_unseen] = 100
# mask_seen = (score_for_seen == 0).cuda()
# preds[mask_seen] = 100

# target_classes = dataset.seenclasses.cuda()
# acc_seen = compute_per_class_acc(util.map_label(dataset.test_seen_label, target_classes), preds, target_classes.size(0))
# print("seen_acc = ", acc_seen)
# print('\n')
# target_classes = dataset.novelclasses.cuda()
# acc_unseen = compute_per_class_acc(util.map_label(dataset.test_unseen_label, target_classes), util.map_label(test_label, target_classes), target_classes.size(0))
# print("unseen_acc = ",acc_unseen)




