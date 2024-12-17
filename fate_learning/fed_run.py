from fate.ml.nn.homo.fedavg import FedAVGArguments, FedAVGClient, FedAVGServer, TrainingArguments
from fate.arch import Context
import torch as t
from fate.arch.launchers.multiprocess_launcher import launch
from sklearn.metrics import roc_auc_score
from torchvision import models, datasets, transforms
from fate.ml.nn.dataset.table import TableDataset

from sampling import dist_datasets_iid, dist_datasets_noniid
from options import args_parser

from models import CNNMnistRelu, CNNMnistTanh
from models import CNNFashion_MnistRelu, CNNFashion_MnistTanh
from models import CNNCifar10Relu, CNNCifar10Tanh
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import numpy as np

import os,sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ))
sys.path.append(root_path)

cur_dir = os.path.join(root_path,'fate_learning')
log_dir = os.path.join(cur_dir,'..','backend', 'result', 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

deepspeed_config = {
    "train_micro_batch_size_per_gpu": 16,
    "train_batch_size": "auto",
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.01
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0
        }
    },
    "fp16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": False,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
    }
}

def auc(pred):
    pred_labels = np.argmax(pred.predictions, axis=1)
    correct = np.sum(np.equal(pred.label_ids, pred_labels))
    total = len(pred_labels)

    with open(log_dir + '/' + 'test.txt', mode='a') as f:
        f.write("auc:%s\n" % (correct/total))
    return {'auc': correct/total}


# 修改二 构造数据集和模型
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone().detach(), torch.tensor(label)

def get_setting(ctx: Context):

    # prepare data
    data_dir = os.path.join(cur_dir, 'dataset', 'minist')
    apply_transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                        transform=apply_transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                transform=apply_transform)
    
    user_groups = dist_datasets_iid(train_dataset, 10)
    u = ctx._federation.local_party[1]

    train_dataset = DatasetSplit(train_dataset, user_groups[int(u)])

    # prepare model
    model = CNNMnistRelu()
    # summary(model, input_size=(1, 28, 28), device='cpu')
            
    # prepare loss
    loss = nn.CrossEntropyLoss()
    # prepare optimizer
    optimizer = t.optim.Adam(model.parameters(), lr=0.001)
    args = TrainingArguments(
        num_train_epochs=30,
        per_device_train_batch_size=256,
        evaluation_strategy='epoch',
    )

    fed_arg = FedAVGArguments(
        aggregate_strategy='epoch',
        aggregate_freq=1
    )

    model.to(device)

    return train_dataset, test_dataset, model, optimizer, loss, args, fed_arg


def train(ctx: Context, 
          train_dataset = None, 
          val_dataset = None, 
          model = None, 
          optimizer = None, 
          loss_func = None, 
          args: TrainingArguments = None, 
          fed_args: FedAVGArguments = None
          ):

    if ctx.is_on_guest or ctx.is_on_host:
        trainer = FedAVGClient(ctx=ctx,
                               model=model,
                               train_set=train_dataset,
                               val_set = val_dataset,
                               optimizer=optimizer,
                               loss_fn=loss_func,
                               training_args=args,
                               fed_args=fed_args,
                               compute_metrics=auc,
                               )
        trainer.train()
    elif ctx.is_on_arbiter:
        trainer = FedAVGServer(ctx)
        trainer.train()

    return trainer

def run(ctx: Context): 

    if ctx.is_on_arbiter:
        train(ctx)
    else:
        train(ctx, *get_setting(ctx))

if __name__ == '__main__':
    launch(run)


