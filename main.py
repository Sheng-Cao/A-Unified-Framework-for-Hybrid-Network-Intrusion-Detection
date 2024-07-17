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
from tqdm import trange
from model import Embedding_Net

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# required functions
def map_label(label, classes):
    mapped_label = torch.zeros_like(label, dtype=torch.long)
    for i, class_label in enumerate(classes):
        mapped_label[label == class_label] = i
    return mapped_label.to(device)


def inverse_map(label, classes):
    mapped_label = np.zeros_like(label)
    classes = classes.cpu().numpy()
    for i, class_label in enumerate(classes):
        mapped_label[label == i] = class_label
    return mapped_label


def indicator(K_matrix, sentry):
    seen_count = (K_matrix < sentry).sum(dim=1)
    unseen_count = K_matrix.shape[1] - seen_count
    score = unseen_count
    return score


def evaluation(y_true, score, option):
    if option == 3:
        report = classification_report(y_true, score,
                                       target_names=dataset.traffic_names[dataset.knownclasses.cpu().numpy()], digits=4)
        print(report)
        return
    if option == 4:
        report = classification_report(y_true, score,
                                       target_names=dataset.traffic_names[dataset.novelclasses.cpu().numpy()], digits=4)
        print(report)
        return
    if option == 5:
        report = classification_report(y_true, score, target_names=dataset.traffic_names, digits=4)
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
            if cls in dataset.knownclasses:
                tp = torch.sum((y_pred == 0) & (dataset.test_label[dataset.benign_size_test:] == cls)).item()
                fn = torch.sum((y_pred == 1) & (dataset.test_label[dataset.benign_size_test:] == cls)).item()
            else:
                tp = torch.sum((y_pred == 1) & (dataset.test_label[dataset.benign_size_test:] == cls)).item()
                fn = torch.sum((y_pred == 0) & (dataset.test_label[dataset.benign_size_test:] == cls)).item()

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recall_per_class[cls.item()] = recall

    print("Recall per class:")
    for cls, recall in recall_per_class.items():
        print(f"Class {dataset.traffic_names[cls]}: Recall = {recall}")
    return y_pred, best_threshold


parser = argparse.ArgumentParser()

# set hyperparameters
parser.add_argument('--dataset', default='181')
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
detector_prediction, _ = evaluation(y_true, detector_score, 0)
print("end evaluating detector")
print("Inference time of the detector：%.4f seconds" % (end_time - start_time))

# 2nd step: Discriminate unknown categories traffic
test_feature = dataset.all_malicious_feature[dataset.train_seen_feature.shape[0]:]

# sentry denotes whether it belongs to the test set or training set
sentry = dataset.train_seen_feature.shape[0]

# initailize hyperparameters and matrixes
pairwise = nn.PairwiseDistance(p=2)
K = 20
indice_matrix = torch.IntTensor(size=(test_feature.shape[0], K)).to(device)

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
discriminator_score = indicator(indice_matrix, sentry)

end_time = time.time()

print("Training and inference time of the discriminator：%.4f seconds" % (end_time - start_time))

y_true = dataset.test_seen_unseen_label.cpu().numpy()

discriminator_score = discriminator_score.cpu().numpy()

discriminator_prediction, threshhold = evaluation(y_true, discriminator_score, 1)
print("end fittting and evaluating discriminator")

# 3rd step: Classify known categories traffic

# training proceduce of known_class_classifier
known_class_classifier = xgb.XGBClassifier()
known_class_classifier.fit(dataset.train_seen_feature.cpu().numpy(),
                           map_label(dataset.train_seen_label, dataset.knownclasses).cpu().numpy())
# inferece proceduce of known_class_classifier
known_preds = known_class_classifier.predict(dataset.test_seen_feature.cpu().numpy())
# evaluation of known_class_classifier
evaluation(map_label(dataset.test_seen_label, dataset.knownclasses).cpu().numpy(), known_preds, 3)

# 4th step: Classify unknown categories traffic (Optional)
# load unknown_class_classifier
unknown_class_classifier = xgb.XGBClassifier()
unknown_class_classifier.load_model('./models/' + opt.dataset + '/cls.model')
mapper = Embedding_Net(opt).to(device)
mapper.load_state_dict(torch.load('./models/' + opt.dataset + '/map.pt'))
mapper.eval()
# inferece proceduce of unknown_class_classifier
with torch.no_grad():
    embed, _ = mapper(dataset.test_unseen_feature)

unknown_preds = unknown_class_classifier.predict(embed.cpu().numpy())
# evaluation of unknown_class_classifier
evaluation(map_label(dataset.test_unseen_label, dataset.novelclasses).cpu().numpy(), unknown_preds, 4)

# 5th step: hybrid ultimate output (Unknown predictions intergration is optional)
# inverse predictions for malicious traffic
known_preds_inverse = inverse_map(known_preds, dataset.knownclasses)
unknown_preds_inverse = inverse_map(unknown_preds, dataset.novelclasses)

# collect all predictions (benign and malicious) and mask misclassified as 100
preds_all = np.concatenate(
    (detector_prediction[:dataset.benign_size_test].cpu().numpy(), known_preds_inverse, unknown_preds_inverse), axis=0)

# get score for benign and malicious traffic
score_for_benign = detector_prediction[:dataset.benign_size_test]
score_for_malicious = detector_prediction[dataset.benign_size_test:]

# get score for known and unknown malicious traffic
score_for_known = discriminator_prediction[:dataset.test_seen_feature.shape[0]]
score_for_unknown = discriminator_prediction[dataset.test_seen_feature.shape[0]:]

# get index for misdetected benign traffic
det_wrong_benign = torch.where(score_for_benign == 1)[0].to(device)
# get index for undetected malicious traffic
det_wrong_malicious = torch.where(score_for_malicious == 0)[0].to(device)

# get index for misdiscriminated known malicious traffic
dis_wrong_known = torch.where(score_for_known == 1)[0].to(device)
# get index for misdiscriminated unknown malicious traffic
dis_wrong_unknown = torch.where(score_for_unknown == 0)[0].to(device)

with torch.no_grad():
    # 1.give misdiscriminated known malicious traffic new labels from unknown classes
    if len(dis_wrong_known) > 0:
        known_features = dataset.test_seen_feature[dis_wrong_known]
        corrected_known_preds = unknown_class_classifier.predict(mapper(known_features)[0].cpu().numpy())
        corrected_known_preds_inverse = inverse_map(corrected_known_preds, dataset.novelclasses)
        preds_all[dataset.benign_size_test + dis_wrong_known.cpu().numpy()] = corrected_known_preds_inverse

    # 2. give misdiscriminated unknown malicious traffic new labels from known classes
    if len(dis_wrong_unknown) > 0:
        unknown_features = dataset.test_unseen_feature[dis_wrong_unknown]
        corrected_unknown_preds = known_class_classifier.predict(unknown_features.cpu().numpy())
        corrected_unknown_preds_inverse = inverse_map(corrected_unknown_preds, dataset.knownclasses)
        preds_all[dataset.benign_size_test + dataset.test_seen_feature.shape[
            0] + dis_wrong_unknown.cpu().numpy()] = corrected_unknown_preds_inverse

    # 3. give misdetected benign traffic new labels from malicious classes
    if len(det_wrong_benign) > 0:
        test_benign_feature = dataset.test_feature[:dataset.benign_size_test]
        benign_features = test_benign_feature[det_wrong_benign]

        indice_matrix = torch.IntTensor(size=(benign_features.shape[0], K)).to(device)
        for i in trange(benign_features.shape[0]):
            expand_feature = benign_features[i].unsqueeze(0).expand_as(dataset.all_malicious_feature)
            dis = pairwise(expand_feature, dataset.all_malicious_feature)
            # sort and selection
            _, indices = torch.topk(dis, k=K, largest=False)
            indice_matrix[i] = indices
        benign_discriminator_scores = indicator(indice_matrix, sentry).cpu().numpy()

        known_unknown_preds = np.where(benign_discriminator_scores < threshhold,
                                       inverse_map(known_class_classifier.predict(benign_features.cpu().numpy()),
                                                   dataset.knownclasses),
                                       inverse_map(
                                           unknown_class_classifier.predict(mapper(benign_features)[0].cpu().numpy()),
                                           dataset.novelclasses))
        preds_all[det_wrong_benign.cpu().numpy()] = known_unknown_preds

    # 4. give undetected malicious traffic benign labels
    if len(det_wrong_malicious) > 0:
        preds_all[dataset.benign_size_test + det_wrong_malicious.cpu().numpy()] = 0

# Final evaluation
evaluation(dataset.test_label.cpu().numpy(), preds_all, 5)
