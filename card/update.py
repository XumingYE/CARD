
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class LocalUpdate(object):
    def __init__(self, args, dataset=None, name=None):
        self.args = args
        self.name = name
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(dataset, batch_size=args.local_bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (input_label, pos_label, neg_label, ngram_label) in enumerate(tqdm(self.ldr_train, desc=f"{self.name} Local Updating: ")):
                input_label = input_label.long().to(self.args.device)
                pos_label = pos_label.long().to(self.args.device)
                neg_label = neg_label.long().to(self.args.device)
                ngram_label = ngram_label.long().to(self.args.device)
                # 3 step in torch
                optimizer.zero_grad()
                loss = net(input_label, pos_label, neg_label, ngram_label).mean()
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(input_label), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)