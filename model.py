from datetime import datetime
import torch
import torch.nn as nn

class Extractor(nn.Module):

    def __init__(self, in_dim=8, out_dim=64):
        super(Extractor, self).__init__()
        self.out_dim = out_dim
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=(32, 1)),
            nn.BatchNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.MaxPool2d(kernel_size=(8, 1), stride=4)

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(16, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer4 = nn.MaxPool2d(kernel_size=(4, 1), stride=2)

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=out_dim, kernel_size=(8, 1)),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        self.layer6 = nn.AdaptiveMaxPool2d((out_dim, 1))

    def forward(self, x):
        x = torch.transpose(x, 2, 3)
        B, L, H, W = x.shape
        x = x.view(B*L, H, W, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(B, L, self.out_dim, self.out_dim)
        return x

class Generator(nn.Module):

    def __init__(self, in_dim, sigma=1.0, adj_ratio=0.5):
        super(Generator, self).__init__()
        self.adj_dim = in_dim
        self.sigma = sigma
        self.adj_ratio = adj_ratio

        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)

    def gaussian_kernel(self, x, y):
        diff = x - y
        dist_sq = torch.sum(diff ** 2, dim=-1)
        return torch.exp(-dist_sq / (2 * self.sigma ** 2))

    def forward(self, feature):
        B, L, H, W = feature.shape
        feature = feature.view(B*L, H, W)

        # Attention-based adjacency matrix
        Q = self.query(feature)
        K = self.key(feature)
        V = self.value(feature)
        attention = torch.bmm(K.transpose(1, 2), Q) / (feature.size(-1) ** 0.5)
        attention = torch.softmax(attention, dim=-1)
        adj_attention = torch.bmm(V, attention)

        # Gaussian kernel-based adjacency matrix
        adj_gaussian = torch.zeros(B*L, self.adj_dim, self.adj_dim).to(feature.device)
        for i in range(self.adj_dim):
            for j in range(self.adj_dim):
                adj_gaussian[:, i, j] = self.gaussian_kernel(feature[:, i, :], feature[:, :, j])


        adj_attention = torch.softmax(adj_attention, dim=-1)
        adj_gaussian = torch.softmax(adj_gaussian, dim=-1)

        # Combine the two adjacency matrices
        adj_combined = adj_attention + adj_gaussian

        # generate the mask
        graph_mask = self.get_mask(adj_combined)
        graph_mask = graph_mask.view(B, L, self.adj_dim, self.adj_dim)
        return graph_mask

    def get_mask(self, adj):
        gumbel = self.sample_gumbel(adj)
        adj += gumbel
        num_elements = int(self.adj_ratio * adj.numel() / adj.size(0))
        topk_values, _ = torch.topk(adj.view(adj.size(0), -1), num_elements, dim=1)
        threshold = topk_values[:, -1].unsqueeze(1).unsqueeze(2).expand_as(adj)
        mask = (adj >= threshold).float()
        return mask

    def sample_gumbel(self, logits, eps=1e-20):
        U = torch.rand(logits.shape).to(logits.device)
        return -torch.log(-torch.log(U + eps) + eps)


class GCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, loop=1):
        super(GCN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.loop = loop

        self.weight = torch.nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = torch.nn.Parameter(torch.FloatTensor(out_dim))
        self.relu = torch.nn.ReLU()

        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, mask):
        B, L, H, W = x.shape
        x = x.view(B*L, H, W)
        mask = mask.view(B*L, H, W)
        x = torch.matmul(torch.matmul(self.d_i_p(mask), x), self.weight) + self.bias
        x = self.relu(x)
        x = x.view(B, L, H, self.out_dim)
        return x

    def d_i_p(self, x):
        x = self.self_loop(x)
        d = torch.sum(x, dim=1)
        D = torch.stack([torch.diag(d_i) for d_i in d])
        I = torch.inverse(D)
        d_i_pow = torch.sqrt(I)
        return d_i_pow

    def self_loop(self, x):
        I = torch.eye(x.size(-1)).to(x.device)
        return x * (1 - I) + self.loop * I


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.lstm = nn.LSTM(input_dim*input_dim, hidden_dim, num_layers=layers, batch_first=True)

    def forward(self, x):
        B, L, H, W = x.shape
        x = x.view(B, L, H*W)
        h0 = torch.zeros(self.layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return out[:, -1, :]


class MPML(nn.Module):

    def __init__(self,
                 class_num = 5,  # class
                 max_size = 100,  # exemplar max size
                 time_step = 10,  # time step
                 signal_num = 8,  # signal num
                 adj_dim=64,  # graph node num
                 gcn_layer=2,  # gcn layer
                 lstm_layer=2,  # lstm layer
                 h_dim=1024,  # lstm hidden layer dim
                 adj_ratio=0.4,  # adj matrix edge ratio
                 gaussian_sigma=1.0,  # gaussian kernel parameter
                 loop=1  # self-loop coefficient
                 ):

        super(MPML, self).__init__()
        self.version = datetime.now().strftime("%Y%m%d%H%M%S")
        self.class_num = class_num
        self.max_size = max_size
        self.time_step = time_step
        self.gcn_layer = gcn_layer
        self.h_dim = h_dim

        self.extractor = Extractor(in_dim=signal_num, out_dim=adj_dim)

        self.generator = Generator(in_dim=adj_dim, sigma=gaussian_sigma, adj_ratio=adj_ratio)

        self.gcn_1 = GCN(in_dim=adj_dim, out_dim=adj_dim, loop=loop)
        self.gcn_2 = GCN(in_dim=adj_dim, out_dim=adj_dim, loop=loop)

        self.lstm =LSTM(input_dim=adj_dim, hidden_dim=h_dim, layers=lstm_layer)

        self.classifier = nn.Linear(h_dim, class_num)

        self.exemplar_means = None
        self.max_size = 100
        self.exemplars = None

    def forward(self, data):  # [B, F, 8]
        data = data.chunk(self.time_step, dim=1)
        feature = torch.stack(data, dim=1)

        feature1 = self.extractor(feature)

        mask = self.generator(feature1)
        feature2 = self.gcn_1(feature1, mask)
        feature3 = self.gcn_2(feature2, mask)

        feature4 = self.lstm(feature3)

        out = self.classifier(feature4)  # [B, C]
        return out, (feature, feature1, feature2, feature3, feature4)

    def extend_classifier(self, new_classes):
        old_weight = self.classifier.weight.data
        old_bias = self.classifier.bias.data
        self.classifier = nn.Linear(self.h_dim, self.classifier.out_features + new_classes).to(self.classifier.weight.device)
        self.classifier.weight.data[:old_weight.size(0)] = old_weight
        self.classifier.bias.data[:old_bias.size(0)] = old_bias
        self.classifier.weight.data[old_weight.size(0):] = torch.randn(new_classes, self.h_dim) * 0.01
        self.classifier.bias.data[old_bias.size(0):] = torch.zeros(new_classes)
        self.class_num += new_classes
        self.version = datetime.now().strftime("%Y%m%d%H%M%S")

    def compute_class_means(self, data_load):
        self.eval()
        with torch.no_grad():
            class_means = {}
            for batch in data_load:
                data = batch[0].to(self.classifier.weight.device)
                label = batch[1].to(self.classifier.weight.device)
                _, (_, _, _, _, feature) = self(data)
                for i in range(data.size(0)):
                    if label[i].item() not in class_means:
                        class_means[label[i].item()] = feature[i].unsqueeze(0)
                    else:
                        class_means[label[i].item()] = torch.cat(
                            [class_means[label[i].item()], feature[i].unsqueeze(0)], dim=0)
            for key in class_means.keys():
                class_means[key] = class_means[key].mean(dim=0)
        self.exemplar_means = class_means
        self.train()

    def update_exemplars(self, data_load):
        self.eval()
        exemplars = {}
        with torch.no_grad():
            for batch in data_load:
                data = batch[0].to(self.classifier.weight.device)
                label = batch[1].to(self.classifier.weight.device)
                _, (_, _, _, _, feature) = self(data)
                for i in range(data.size(0)):
                    if label[i].item() not in exemplars:
                        exemplars[label[i].item()] = feature[i].unsqueeze(0)
                    else:
                        exemplars[label[i].item()] = torch.cat([exemplars[label[i].item()], feature[i].unsqueeze(0)], dim=0)
            for key in exemplars.keys():
                if key in self.exemplar_means.keys():
                    exemplars[key] = torch.cat([exemplars[key], self.exemplar_means[key].unsqueeze(0)], dim=0)
                if exemplars[key].size(0) > self.max_size:
                    exemplars[key] = exemplars[key][:self.max_size]
        self.exemplars = exemplars
        self.train()


if __name__ == "__main__":
    x = torch.randn(8, 25600, 8)
    model = MPML()
    o = model(x)
    print(o.shape)
