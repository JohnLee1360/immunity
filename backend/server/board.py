from flask import (
    Blueprint, flash, g, redirect,  request, jsonify, url_for, Response, stream_with_context
)
bp = Blueprint('board', __name__, url_prefix='/board')

import sys, os, subprocess, time, threading, json
#immunity 根目录
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
atk_log = os.path.join(root_path,'fate_learning','atk.txt')

process = None
process_status = "idle"

def fed_run():
    global process_status, root_path
    fed_script = os.path.join(root_path, 'fate_learning', 'fed_run.py')
    print(fed_script)
    process = subprocess.Popen(
        ['python3', fed_script , '--parties', 'arbiter:0','guest:0', 'host:1', 'host:2','host:3', '--log_level', 'INFO'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True
    )
    process_status = "running"
    print("running above script")

    process.wait()
    process_status = "stopped"

    if process.poll is not None:
        print("Finished!")


@bp.route('/train', methods = ['GET', 'POST'])
def train():
    global process, process_status, root_path
    if request.method == 'POST':
        #检查时候否有正在运行的任务
        if process and process_status == "running":
            return jsonify({"message" : "Training process is already running."}),400
        # 启动新线程执行fed_run，防止阻塞住进程
        process_thread = threading.Thread(target=fed_run)
        process_thread.start()
        return jsonify({"message" : "Training process started."}),200
    if request.method == 'GET':
        return process_status

@bp.route('/train-status', methods = ['GET', 'POST'] )
def train_status():
    def generate():
        while True:
            if process_status == "idle":
                yield f"data: {json.dumps({'status': 'idle', 'attacker': None})}\n\n"
                break 

            if process_status == "running":
                time.sleep(2)
                try:
                    with open(atk_log, 'r') as file:
                        lines = file.readlines()
                        if lines:
                            file.seek(file.tell() - 2, os.SEEK_SET)  # 移动到最后一个字符
                            attacker = file.read(1).strip()  # 读取最后一个字符
                        else:
                            attacker = None
                        yield f"data: {json.dumps({'status': 'running', 'attacker': attacker})}\n\n"
                except FileNotFoundError:
                    attacker = None
                    yield f"data: {json.dumps({'status': 'stopped', 'attacker': attacker})}\n\n"
                    break

            elif process_status == "stopped":
                yield f"data: {json.dumps({'status': 'stopped', 'attacker': None})}\n\n"
                break  # 结束推送
            
            time.sleep(1)  # 减少轮询时间

    #设置header以保持连接
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
    }
    return Response(generate(), headers=headers)
train_status()



    