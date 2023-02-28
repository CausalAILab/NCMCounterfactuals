import pytorch_lightning as pl
import torch as T
from torch.utils.data import DataLoader, Dataset


class BasePipeline(pl.LightningModule):
    min_delta = 1e-6
    patience = 20
    max_epochs = 10000

    def __init__(self, generator, do_var_list, dat_sets, cg, dim, ncm, batch_size=256):
        super().__init__()
        self.generator = generator
        self.do_var_list = do_var_list
        self.dat_sets = dat_sets
        self.cg = cg
        self.ncm = ncm
        self.dim = dim

        self.batch_size = batch_size

        self.stored_metrics = None

    def forward(self, n=1, u=None, do={}):
        return self.ncm(n, u, do)

    def train_dataloader(self):
        return DataLoader(SCMDataset(self.dat_sets),
            batch_size=self.batch_size, shuffle=True, drop_last=True)

    def update_metrics(self, new_metrics):
        self.stored_metrics = new_metrics


class SCMDataset(Dataset):
    def __init__(self, dat_sets):
        self.dat_sets = dat_sets
        self.length = len(dat_sets[0][next(iter(dat_sets[0]))])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return [{k: self.dat_sets[i][k][idx] for k in self.dat_sets[i]} for i in range(len(self.dat_sets))]
