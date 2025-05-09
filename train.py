import copy
import os
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import trange
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch.nn.functional as F

from model import MPML


class Dataset_my(Dataset):
    def __init__(self, _data, _label):
        self.len = len(_data)
        self.data = _data.astype(np.float32)
        self.label = _label.astype(np.int64)

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return self.len


def load_dataset(dir_path: str, cut_length=12800, overlap_ratio=0.5):
    data_list, label_list = [], []
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        data = load_data(file_path, cut_length, overlap_ratio)
        label = int(os.path.basename(file).split('.')[0])
        label = np.array([label] * len(data))
        data_list.append(data)
        label_list.append(label)
    data = np.concatenate(data_list)
    label = np.concatenate(label_list)
    return Dataset_my(data, label)


def load_all_data(dir_path: str, cut_length=12800, overlap_ratio=0.5):
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        data, label = load_data(file_path, cut_length, overlap_ratio)
        yield data, label


def load_data(file, cut_length, overlap_ratio):
    """
    Load csv data and cut it into segments
    """
    data = pd.read_csv(file)
    data = cut_data(data, cut_length, overlap_ratio)
    return data


def cut_data(data, cut_length: int, overlap_ratio: float):
    data = data.values  # Convert DataFrame to NumPy array
    data_length = len(data)
    overlap_length = int(cut_length * overlap_ratio)
    step = cut_length - overlap_length
    num_segments = (data_length - cut_length) // step + 1

    cut_data_list = [data[i * step:i * step + cut_length] for i in range(num_segments)]
    return np.array(cut_data_list)


class Trainer:

    def __init__(self,
                 data_path,
                 model_path,
                 batch_size=64,
                 cut_length=2560,
                 overlap_ratio=0.5,
                 class_num=4,
                 time_step=10,
                 signal_num=6,
                 adj_dim=64,
                 gcn_layer=2,
                 lstm_layer=2,
                 h_dim=1024,
                 adj_ratio=0.5,
                 gaussian_sigma=1.0,
                 loop=1.0,
                 lr=0.0001,
                 betas=(.5, .9),
                 output_path='output/out',
                 pf=None,
                 ):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_path = output_path
        self.pf = pf
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if not os.path.exists(self.output_path + '/tsne'):
            os.makedirs(self.output_path + '/tsne')
        dataset = load_dataset(data_path, cut_length=cut_length, overlap_ratio=overlap_ratio)
        self.dataset = dataset
        train_dataset, test_dataset = train_test_split(dataset, test_size=0.3, stratify=dataset.label)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if model_path is None or model_path == '':
            self.model = MPML(class_num=class_num, time_step=time_step, signal_num=signal_num, adj_dim=adj_dim,
                              gcn_layer=gcn_layer, lstm_layer=lstm_layer, h_dim=h_dim, adj_ratio=adj_ratio,
                              gaussian_sigma=gaussian_sigma, loop=loop).to(self.device)
            print('create new model')
        else:
            self.model = torch.load(model_path, map_location=self.device)
            print(f'load model from: {model_path}')

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.kd_loss = torch.nn.KLDivLoss()

        self.loss_list = []
        self.acc_list = []

    def train(self, iterations=100):
        # torch.autograd.set_detect_anomaly(True)
        starting_time = time.time()
        self.pf.append(f'Start Training at   :  {time.strftime("%H:%M:%S", time.localtime(starting_time))}\n'
              f'Device              :  {self.device}\n'
              f'Output path         :  {self.output_path}\n'
              f'Starting ...\n')
        for epoch in trange(iterations, desc='Training', unit='epoch'):
            self.model.train()
            train_loss = 0
            batch_num = 0
            for batch in self.train_loader:
                # print(f'iter: {epoch}/{iterations}  batch: {batch_num}/{len(self.train_loader)}')
                batch_num += 1
                data = batch[0].to(self.device)  # [B, 12800, 8]
                label = batch[1].to(self.device)  # [B]
                out, _ = self.model(data)
                loss = self.ce_loss(out, label)
                train_loss += loss.item()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.scheduler.step()
            self.loss_list.append(train_loss)
            self.acc_list.append(self.evaluate())
            self.pf.append(f'#####> iter: {epoch}   loss: {self.loss_list[-1]:.3f}   acc {self.acc_list[-1]:.3f}\n')

        self.save()
        self.pf.append(f"Training took: {time.time() - starting_time:.2f} seconds\n")

        # self.confusion_matrix()
        # self.tsne()

    def train_inc(self, new_classes, iterations=100):
        self.old_model = copy.deepcopy(self.model)
        self.old_model.to(self.device)
        self.model.extend_classifier(new_classes)
        self.optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=0.0001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        for epoch in range(iterations):
            self.model.train()
            train_loss = 0
            for label in self.model.exemplars:
                data = self.model.exemplars[label].to(self.device)
                label = torch.tensor([label] * len(data)).to(self.device)
                out = self.model.classifier(data)
                out_old = self.old_model.classifier(data)
                ce = self.ce_loss(out, label)
                kd = F.kl_div(F.log_softmax(out[:, :out_old.size(1)], dim=1),
                              F.softmax(out_old, dim=1), reduction='batchmean')
                loss = ce + kd
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            for batch in self.train_loader:
                data = batch[0].to(self.device)
                label = batch[1].to(self.device)
                out, _ = self.model(data)
                out_old, _ = self.old_model(data)
                ce = self.ce_loss(out, label)
                kd = F.kl_div(F.log_softmax(out[:, :out_old.size(1)], dim=1),
                              F.softmax(out_old, dim=1), reduction='batchmean')
                loss = ce + kd
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.loss_list.append(train_loss)
            self.acc_list.append(self.evaluate())

            self.pf.append(f'#####> iter: {epoch}   loss: {self.loss_list[-1]:.3f}   acc {self.acc_list[-1]:.3f}\n')

        self.save()

    def save(self):
        with open(self.output_path + '/loss.txt', 'w') as f:
            for i in self.loss_list:
                f.write(str(i) + '\n')
        with open(self.output_path + '/acc.txt', 'w') as f:
            for i in self.acc_list:
                f.write(str(i) + '\n')

        # self.model.compute_class_means(self.train_loader)
        # self.model.update_exemplars(self.train_loader)
        torch.save(self.model, self.output_path + '/model.pt')

    def evaluate(self):
        acc = 0
        total = 0
        self.model.eval()
        for batch in self.test_loader:
            data = batch[0].to(self.device)
            label = batch[1].to(self.device)
            out, _ = self.model(data)
            _, preds = out.max(1)
            acc += (preds == label).sum().item()
            total += len(label)

        acc = acc / total
        return acc

    def eval_incr(self):
        self.confusion_matrix()
        self.tsne()

    def confusion_matrix(self):
        print("confusion matrix...")
        confusion_matrix = np.zeros((self.dataset.label.max() + 1, self.dataset.label.max() + 1))  # 横轴为预测，纵轴为真实
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                data = batch[0].to(self.device)
                label = batch[1].to(self.device)
                out, _ = self.model(data)
                _, preds = out.max(1)

                for i in range(len(label)):
                    confusion_matrix[label[i], preds[i]] += 1
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True) * 100
        np.savetxt(self.output_path + '/confusion_matrix.txt', confusion_matrix, fmt='%d', delimiter=',')

        print("acc, precision, recall, f1...")
        acc = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
        precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
        recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        f1 = 2 * precision * recall / (precision + recall)
        with open(self.output_path + '/acc_precision_recall_f1.txt', 'w') as f:
            f.write(f"Class    acc     precision    recall     f1\n")
            for i in range(len(precision)):
                f.write(f"{i}\t{acc:.3f}\t{precision[i]:.3f}\t{recall[i]:.3f}\t{f1[i]:.3f}\n")
            f.write(f"Average\t{acc:.3f}\t{np.mean(precision):.3f}\t{np.mean(recall):.3f}\t{np.mean(f1):.3f}\n")

    def tsne(self):
        print("t-sne...")
        f0, f1, f2, f3, f4, o = [], [], [], [], [], []
        labels = []
        with torch.no_grad():
            for batch in self.test_loader:
                data = batch[0].to(self.device)
                label = batch[1].to(self.device)
                out, (fe, fe1, fe2, fe3, fe4) = self.model(data)
                f0.append(fe)
                f1.append(fe1)
                f2.append(fe2)
                f3.append(fe3)
                f4.append(fe4)
                o.append(out)
                labels.append(label)

        tsne = TSNE(n_components=2, random_state=42)

        save_to_file(f0, tsne, self.output_path + '/tsne/f0.txt')
        save_to_file(f1, tsne, self.output_path + '/tsne/f1.txt')
        save_to_file(f2, tsne, self.output_path + '/tsne/f2.txt')
        save_to_file(f3, tsne, self.output_path + '/tsne/f3.txt')
        save_to_file(f4, tsne, self.output_path + '/tsne/f4.txt')
        save_to_file(o, tsne, self.output_path + '/tsne/f5.txt')

        labels = torch.cat(labels, dim=0).cpu().numpy()
        with open(self.output_path + '/tsne/labels.txt', 'w') as f:
            for i in labels:
                f.write(str(i) + '\n')


def save_to_file(data, tsne, filename):
    data = torch.cat(data, dim=0).cpu().numpy()
    if data.ndim > 2:
        data = data.reshape(data.shape[0], -1)
    t_d = tsne.fit_transform(data)
    with open(filename, 'w') as f:
        for i in t_d:
            f.write(str(i[0]) + ',' + str(i[1]) + '\n')


class Predictor:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, data):
        with torch.no_grad():
            data = torch.tensor(data).float().to(self.model.classifier.weight.device)
            out, _ = self.model(data)
            _, preds = out.max(1)
            return preds.cpu().numpy()

if __name__ == "__main__":
    path = '../autodl-tmp/d2'
    # path = 'data/test'
    model_path = None
    iters = 100
    train = Trainer(data_path=path, model_path=model_path, output_path='output2/out_2', lr=0.0001)
    train.train(iters)

    # path = '../autodl-tmp/d2'
    # model_path = 'output/out_8/model_.pt'
    # train = Trainer(data_path=path, model_path=model_path, output_path='output2/out_1',lr=0.0001)
    # train.eval_incr()
