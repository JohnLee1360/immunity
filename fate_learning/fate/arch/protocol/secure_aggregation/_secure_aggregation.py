#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import typing
import os
import pandas as pd
import numpy as np
import copy
import pickle
from collections import Counter

import numpy
from fate.arch import Context
from fate.arch.protocol.diffie_hellman import DiffieHellman
from fate_utils.secure_aggregation_helper import MixAggregate, RandomMix

import torch
from torchvision import models, datasets, transforms
from torch import nn
from torch.utils.data import DataLoader

import logging, math

class _SecureAggregatorMeta:
    _send_name = "mixed_client_values"
    _recv_name = "aggregated_values"
    prefix: str

    def _get_name(self, name):
        if self.prefix:
            return f"{self.prefix}_{name}"
        return name


class SecureAggregatorClient(_SecureAggregatorMeta):
    def __init__(self, prefix: typing.Optional[str] = None, is_mock: bool = False):
        """
        secure aggregation client
        Args:
            prefix: unique prefix for this aggregator
            is_mock: mock the aggregator, do not perform secure aggregation, for test only
        """
        self.prefix = prefix
        self._mixer = None
        self._is_mock = is_mock
        self.public_dataset = None

    def _get_mixer(self):
        if self._mixer is None:
            raise RuntimeError("mixer not initialized, run dh_exchange first")
        return self._mixer

    def dh_exchange(self, ctx: Context, ranks: typing.List[int]):
        if self._is_mock:
            return
        local_rank = ctx.local.rank
        dh = {}
        seeds = {}
        for rank in ranks:
            if rank == local_rank:
                continue
            dh[rank] = DiffieHellman()
            ctx.parties[rank].put(self._get_name(f"dh_pubkey"), dh[rank].get_public_key())
        for rank in ranks:
            if rank == local_rank:
                continue
            public_key = ctx.parties[rank].get(self._get_name(f"dh_pubkey"))
            seeds[rank] = dh[rank].diffie_hellman(public_key)
        self._mixer = RandomMix(seeds, local_rank)

    def loss_aggregate(self, ctx: Context, array: typing.List[numpy.ndarray], weight: typing.Optional[int] = None):
        if self._is_mock:
            ctx.arbiter.put(self._get_name(self._send_name), (array, weight))
            return ctx.arbiter.get(self._get_name(self._recv_name))
        else:
            mixed = self._get_mixer().mix(array, weight)
            ctx.arbiter.put(self._get_name(self._send_name), (mixed, weight))
            return ctx.arbiter.get(self._get_name(self._recv_name))

    def secure_aggregate(self, model, ctx: Context, array: typing.List[numpy.ndarray], weight: typing.Optional[int] = None, cur_epoch: typing.Optional[int] = None):
        if self._is_mock:
            ctx.arbiter.put(self._get_name(self._send_name), (array, weight))
            return ctx.arbiter.get(self._get_name(self._recv_name))
        else:
            mixed = self._get_mixer().mix(array, weight)

            # 保存mixed参数
            u = ctx._federation.local_party[1]
            weight_layer0s = np.array(mixed[0])
            cur_dir = os.getcwd()
            model_dir = os.path.join(cur_dir, 'result', 'mixd', u)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_path = model_dir + '/' + str(cur_epoch) + '.csv'
            weight_layer0s = weight_layer0s.reshape(1,-1)
            pd.DataFrame(weight_layer0s).to_csv(model_path, header=False, index=False)

            # zkml
            # mix0 = [mixed[i][0] for i in range(len(mixed))]
            # if self.public_dataset is None:
            #     self.public_dataset = self.get_public_dataset()
            # A1 = self.A_inference(model, array)
            # A2 = self.A_inference(model, mix0)

            # ctx.arbiter.put(self._get_name(self._send_name), (mixed, weight, A1, A2, cur_epoch))



            ctx.arbiter.put(self._get_name(self._send_name), (mixed, weight, cur_epoch))
            return ctx.arbiter.get(self._get_name(self._recv_name))
        

    # 获取公共数据集    
    def get_public_dataset(self):
        data_dir = '../../datasets/mnist/'
        apply_transform = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (0.5,))])
        public_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                    transform=apply_transform)
        return public_dataset
    
    # 生成准确率
    def A_inference(self, model, array):
        public_dataset = self.public_dataset
        modelA = copy.deepcopy(model)
        weight = copy.deepcopy(model.state_dict())
        for i, key in enumerate(weight.keys()):
            weight[key] = torch.tensor(array[i])

        modelA.load_state_dict(weight)
        device = 'cuda:0' 
        modelA.to(device)
        modelA.eval()
        total, correct = 0.0, 0.0     
        Aloader = DataLoader(public_dataset, batch_size=32,
                                shuffle=False)
        for batch_idx, (images, labels) in enumerate(Aloader):
            images, labels = images.to(device), labels.to(device)
            outputs = modelA(images)
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy
    
    # 加密器
    # def encipher(self, ):


class SecureAggregatorServer(_SecureAggregatorMeta):
    def __init__(self, ranks, prefix: typing.Optional[str] = None, is_mock: bool = False):
        """
        secure aggregation serve
        Args:
            ranks: all ranks
            prefix: unique prefix for this aggregator
            is_mock: mock the aggregator, do not perform secure aggregation, for test only
        """
        self.prefix = prefix
        self.ranks = ranks
        self._is_mock = is_mock

    # 检测定位
    def detection_difference(self, node_num = None, epochs = None, dir_ = './result/to_send', k = 5, fusion_node = None, tresh = None): 
        # dir_为'./result/to_send' (reult文件夹是fate_learning跑出来的结果)
        all_datas = []
        for n in range(node_num):
            node_dir = dir_ + '/' + str(n)
            one_datas = []
            for epoch in range(1,epochs+1):
                epoch_path = node_dir + '/' + str(epoch) + '.csv'
                epoch_data = pd.read_csv(epoch_path, header=None, index_col=None).iloc[:].values.flatten()
                one_datas.append(epoch_data)
            all_datas.append(one_datas)

        # nn为融合节点，收集其邻居节点的数据
        all_datas = np.array(all_datas)
        all_datas = all_datas.transpose(1,0,2)

        nn = fusion_node #融合节点     
        # 除去融合节点其他节点数据以及均值
        cur_dir = os.getcwd()
        result_dir = os.path.join(cur_dir,'result/sd_result/process_data',str(nn))
        result_dir = result_dir + '/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        d_difference_data = []
        l_difference_data = []

        no_fusion_datas = np.delete(all_datas, nn, axis=1)
        Ni_mean = np.mean(no_fusion_datas, axis=1)
        
        # 计算𝑔𝑗𝑘(𝑡) − 𝑔̅𝑖𝑘(𝑡)并保存到d_difference_data
        for i in range(node_num):
            d_difference_data.append(all_datas[:,i,:]-Ni_mean)     
            l_difference_data.append(all_datas[:,i,:]-all_datas[:,nn,:])                
        
        d_difference_sum = np.sum(d_difference_data, axis=1) 
        l_difference_sum = np.sum(l_difference_data, axis=1) - d_difference_sum[nn]
        d_difference_sum = d_difference_sum.T
        l_difference_sum = l_difference_sum.T

        d_difference_csv_dir = result_dir + 'layer0_difference_sum_detection_0.csv'
        l_difference_csv_dir = result_dir + 'layer0_difference_sum_localization_0.csv'
        pd.DataFrame(d_difference_sum).to_csv(d_difference_csv_dir, header = None, index = False)
        pd.DataFrame(l_difference_sum).to_csv(l_difference_csv_dir, header = None, index = False)

        # 这里得到的是X_ij，因为前面保存的fineij的值，根据X_ij的公式（1/k ∑k fineij）
        # l_difference_sum = np.delete(l_difference_sum, nn, axis=1)
        d_difference_sum = abs(d_difference_sum)
        l_difference_sum = abs(l_difference_sum)

        # attack
        # detection metric
        d_metric = []
        k_num = d_difference_sum.shape[0] // k  
        for i in range(k_num):
            x_2 = np.sum(d_difference_sum[i*k : (i+1)*k], axis=0, keepdims = True) / k
            result = np.mean(x_2, axis = 1) 
            d_metric.append(result)

        d_metric_csv_dir = result_dir + str(k) + '_metric_detection.csv'
        pd.DataFrame(d_metric).to_csv(d_metric_csv_dir, header = None, index = False)

        # localization metric
        l_metric = []
        for i in range(l_difference_sum.shape[0] // k):
            #按照公式K个参数的值叠加
            z = np.sum(l_difference_sum[i*k:(i+1)*k,:], axis = 0, keepdims = True) / k 
            l_metric.append(z.flatten())

        l_metric_csv_dir = result_dir + str(k) + '_metric_localization.csv'
        pd.DataFrame(l_metric).to_csv(l_metric_csv_dir, header = None, index = False)

        d_metric = np.array(d_metric)
        l_metric = np.array(l_metric)

        # detection
        tresh = 0.04 + 0.01 * math.log(node_num) + 0.04 * (1 - 0.9**(epochs))
        if ((d_metric > tresh).mean()) > 0.9:
            arr = np.argmax(l_metric, axis=1)
            count = Counter(arr)
            attacker_idx = count.most_common(1)[0][0]

            print ("检测到存在攻击者:用户{}".format(attacker_idx))
            return attacker_idx
        # _attacker_idx = None
        # if ((d_metric > tresh).mean()) > 0.9:
        #     arr = np.argmax(l_metric, axis=1)
        #     count = Counter(arr)
        #     _attacker_idx = count.most_common(1)[0][0]
        #     print ("提示:检测到存在可疑攻击者client{}".format(_attacker_idx))
        #     run_count = 0
        #     run_count += 1
        #     if self.run_count >= 5:
        #         print("警告:client{}被判定为内部攻击者！".format(_attacker_idx))
        #         attacker_idx = _attacker_idx
        #     return _attacker_idx, attacker_idx
        # return _attacker_idx, None # 如果没有检测到攻击者的返回值

    def secure_aggregate(self, ctx: Context, ranks: typing.Optional[int] = None):
        """
        perform secure aggregate once
        Args:
            ctx: Context to use
            ranks: ranks to aggregate, if None, use all ranks
        """
        if ranks is None:
            ranks = self.ranks
        aggregated_weight = 0.0
        has_weight = False

        if self._is_mock:
            aggregated = []
            for rank in ranks:
                arrays, weight = ctx.parties[rank].get(self._get_name(self._send_name))
                for i in range(len(arrays)):
                    if len(aggregated) <= i:
                        aggregated.append(arrays[i])
                    else:
                        aggregated[i] += arrays[i]
                if weight is not None:
                    has_weight = True
                    aggregated_weight += weight
            if has_weight:
                aggregated = [x / aggregated_weight for x in aggregated]
        else:
            mix_aggregator = MixAggregate()
            for rank in ranks:
                mix_arrays, weight, cur_epoch = ctx.parties[rank].get(self._get_name(self._send_name))
                # mix_arrays, weight, A1, A2, cur_epoch = ctx.parties[rank].get(self._get_name(self._send_name))

                # u = ctx._federation.local_party[1] 
                # if A1 < 0.10 + cur_epoch*0.01: 
                #     print("存在垃圾节点：{}".format(rank) )  

                mix_aggregator.aggregate(mix_arrays)
                if weight is not None:
                    has_weight = True
                    aggregated_weight += weight

            u = ctx._federation.local_party[1]
            num = len(ranks)
            # 获得ATTACKER
            ATTACKER = self.detection_difference(node_num = num, epochs = cur_epoch, dir_ = './result/to_send', k = 5, fusion_node = int(u), tresh = 0.05)
            print(f"打印到atk_log的值：{ATTACKER}，当前epoch：{cur_epoch}")

            # 输出日志atk.txt
            fl_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
            atk_log = fl_dir + '/atk.txt'
            logging.basicConfig(
                filename=atk_log,
                level=logging.INFO,
                force=True  # 强制重新配置日志
            )
            
            if ATTACKER is not None and (10-cur_epoch):
                logging.info(f'{ATTACKER}')
                logging.getLogger().handlers[0].flush()
            
            if 30-cur_epoch==0:
                logging.info(f'end')
                logging.getLogger().handlers[0].flush()
            

            if not has_weight:
                aggregated_weight = None
            aggregated = mix_aggregator.finalize(aggregated_weight)

        for rank in ranks:
            ctx.parties[rank].put(self._get_name(self._recv_name), aggregated)

        return aggregated


    def loss_aggregate(self, ctx: Context, ranks: typing.Optional[int] = None):
        """
        perform secure aggregate once
        Args:
            ctx: Context to use
            ranks: ranks to aggregate, if None, use all ranks
        """
        if ranks is None:
            ranks = self.ranks
        aggregated_weight = 0.0
        has_weight = False

        if self._is_mock:
            aggregated = []
            for rank in ranks:
                arrays, weight = ctx.parties[rank].get(self._get_name(self._send_name))
                for i in range(len(arrays)):
                    if len(aggregated) <= i:
                        aggregated.append(arrays[i])
                    else:
                        aggregated[i] += arrays[i]
                if weight is not None:
                    has_weight = True
                    aggregated_weight += weight
            if has_weight:
                aggregated = [x / aggregated_weight for x in aggregated]
        else:
            mix_aggregator = MixAggregate()
            for rank in ranks:
                mix_arrays, weight = ctx.parties[rank].get(self._get_name(self._send_name))
                mix_aggregator.aggregate(mix_arrays)
                if weight is not None:
                    has_weight = True
                    aggregated_weight += weight
            if not has_weight:
                aggregated_weight = None
            aggregated = mix_aggregator.finalize(aggregated_weight)

        for rank in ranks:
            ctx.parties[rank].put(self._get_name(self._recv_name), aggregated)

        return aggregated
