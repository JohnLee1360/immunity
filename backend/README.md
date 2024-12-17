## 介绍
Flask后端框架

## 启动
1. 安装依赖的环境:`conda env create -f conda_fl.yml`
```
conda create -n fl python=3.10
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt
```
2. 确保fate_learning/ 文件夹下有一个atx.txt文件
3. 启动后端服务器: `flask --app server run --debug`
4. 打开http://127.0.0.1:5000

## 主要文件
/server/board.py    看板页面后端代码