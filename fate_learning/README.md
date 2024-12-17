## get started
1. initialize fate_flow service and fate_client (optional)

```sh
mkdir fate
fate_flow init --ip 127.0.0.1 --port 9380 --home $(pwd)/fate
pipeline init --ip 127.0.0.1 --port 9380

fate_flow start
fate_flow status # make sure fate_flow service is started

#shut down fate flow if you finish
fate_flow stop
```

2. create an `atx.txt` in this file

3. run the **fed_run.py**
```sh
python fed_run.py --parties arbiter:0 guest:0 host:1 host:2 host:3 --log_level INFO
```

## main files
1. fate_learning/fate/ml/aggregator/base.py   # 注入攻击模型
2. fate_learning/fate/arch/protocol/secure_aggregation/_secure_aggregation.py     # 嵌入检测模型