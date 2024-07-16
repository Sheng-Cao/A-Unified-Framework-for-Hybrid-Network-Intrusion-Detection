import numpy as np
import torch
from sklearn import preprocessing


class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self, opt):
        # read malicious traffic and benign traffic
        matcontent = np.load(f"/home/wy/code/CE-GZSL/data/{opt.dataset}.npz",
                             allow_pickle=True)  # change to your root path
        benign_matcontent = np.load("./data/benign.npz", allow_pickle=True)  # change to your root path

        # benign traffic feature and label
        bx_train = benign_matcontent['bx_train']
        by_train = benign_matcontent['by_train']
        bx_test = benign_matcontent['bx_test']
        by_test = benign_matcontent['by_test']

        # malicious traffic feature and label
        train_feature = matcontent['train_feature']
        train_label = matcontent['train_label'] + 1
        test_seen_feature = matcontent['test_seen_feature']
        test_seen_label = matcontent['test_seen_label'] + 1
        test_unseen_feature = matcontent['test_unseen_feature']
        test_unseen_label = matcontent['test_unseen_label'] + 1
        attacks = matcontent['attacks']

        # preprocessing
        detector_scaler = preprocessing.StandardScaler() if opt.standardization else preprocessing.MinMaxScaler()
        discriminator_scaler = preprocessing.StandardScaler() if opt.standardization else preprocessing.MinMaxScaler()
        classifier_scaler = preprocessing.StandardScaler() if opt.standardization else preprocessing.MinMaxScaler()

        combined_train_feature = np.concatenate((bx_train, train_feature), axis=0)
        combined_test_feature = np.concatenate((bx_test, test_seen_feature, test_unseen_feature), axis=0)

        combined_train_feature = detector_scaler.fit_transform(combined_train_feature)
        combined_test_feature = detector_scaler.transform(combined_test_feature)

        # all features for training detector and inference(including benign traffic features)
        self.train_feature = torch.tensor(combined_train_feature, dtype=torch.float32).cuda()
        self.test_feature = torch.tensor(combined_test_feature, dtype=torch.float32).cuda()

        # all lables for training detector(including benign traffic features)
        self.train_label = torch.tensor(np.concatenate((by_train, train_label), axis=0), dtype=torch.long).cuda()
        self.test_label = torch.tensor(np.concatenate((by_test, test_seen_label, test_unseen_label), axis=0),
                                       dtype=torch.long).cuda()

        # all binary lables for training detector(including benign traffic features, 0 for benign and 1 for malicious)
        binary_by_train = torch.zeros(by_train.shape[0], dtype=torch.long).cuda()
        binary_by_test = torch.zeros(by_test.shape[0], dtype=torch.long).cuda()
        binary_ay_train = torch.ones(train_label.shape[0], dtype=torch.long).cuda()
        binary_ay_test = torch.ones(test_seen_label.shape[0] + test_unseen_label.shape[0], dtype=torch.long).cuda()
        self.binary_train = torch.cat((binary_by_train, binary_ay_train), dim=0)
        self.binary_test = torch.cat((binary_by_test, binary_ay_test), dim=0)

        # all malicious traffic features for training discriminator and inference
        combined_all_malicious_feature = np.concatenate((train_feature, test_seen_feature, test_unseen_feature), axis=0)
        self.all_malicious_feature = torch.tensor(discriminator_scaler.fit_transform(combined_all_malicious_feature),
                                                  dtype=torch.float32).cuda()

        # all binary malicious traffic lables for training discriminator(0 for known and 1 for unknown)
        binary_ky_test = torch.zeros(test_seen_label.shape[0], dtype=torch.long).cuda()
        binary_uy_test = torch.ones(test_unseen_label.shape[0], dtype=torch.long).cuda()
        self.test_seen_unseen_label = torch.cat((binary_ky_test, binary_uy_test), dim=0)

        # all malicious traffic features for training classifier
        self.train_seen_feature = torch.tensor(classifier_scaler.fit_transform(train_feature),
                                               dtype=torch.float32).cuda()
        self.test_seen_feature = torch.tensor(classifier_scaler.transform(test_seen_feature),
                                              dtype=torch.float32).cuda()
        self.test_unseen_feature = torch.tensor(classifier_scaler.transform(test_unseen_feature),
                                                dtype=torch.float32).cuda()

        # all malicious traffic lables for training detector
        self.train_seen_label = torch.tensor(train_label, dtype=torch.long).cuda()
        self.test_seen_label = torch.tensor(test_seen_label, dtype=torch.long).cuda()
        self.test_unseen_label = torch.tensor(test_unseen_label, dtype=torch.long).cuda()

        # known, unknown, malicious, all traffic categories
        self.seenclasses = torch.unique(self.test_seen_label)
        self.novelclasses = torch.unique(self.test_unseen_label)
        self.maliciousclasses = torch.cat((self.seenclasses, self.novelclasses), dim=0)
        self.allclasses = torch.unique(self.test_label)

        # number of samples in training set and test set
        self.ntrain = self.train_feature.size()[0]
        self.ntest = self.test_feature.size()[0]
        # size of benign traffic
        self.benign_size_test = by_test.shape[0]
        # malicious traffic categories
        self.attacks = attacks

        all_labels = torch.cat((self.train_label, self.test_label), dim=0)
        unique_labels, counts = torch.unique(all_labels, return_counts=True)
        # print details of the dataset
        print("Dataset details")
        for i, count in zip(unique_labels, counts):
            print(f'{self.attacks[i]}: {count.item()} samples')
        print("Known malicious traffic categories:", self.attacks[self.seenclasses.cpu()])
        print("Unknown malicious traffic categories:", self.attacks[self.novelclasses.cpu()])