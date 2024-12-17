## 搭链
生成一条单群组4节点的FISCO链。 请确保机器的30300~30303，20200~20203，8545~8548端口没有被占用。
`bash build_chain.sh -l 127.0.0.1:4 -p 30300,20200,8545`

## 启动节点
bash nodes/127.0.0.1/start_all.sh

## 通信协议
注意client_config.py文件,使用rpc协议通信
