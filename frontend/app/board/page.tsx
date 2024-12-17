'use client'

import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"

import { useEffect, useState, useCallback } from "react"
import axios from 'axios'
import { Power } from "lucide-react"

import InfoPanel from "@/components/board-info"

type NodeStatus = 'idle' | 'running' | 'warning' | 'error' | 'stopped'
type NodeType = "0" | "1" | "2" | "3"
interface Node {
  id: NodeType
  label: string
  x: number
  y: number
  status: NodeStatus
}
interface Connection {
  from: NodeType
  to: NodeType
  isActive: boolean
}

export default function Board() {

//初始化
//0:Arbiter  1:Client1  2:Client2  3:Client3
  const [trainingStarted, setTrainingStarted] = useState(false);
  const [nodes, setNodes] = useState<Node[]>([
    { id: "0", label: "Arbiter", x: 250, y: 250, status: "idle" },
    { id: "1", label: "Client1", x: 250, y: 100, status: "idle" },
    { id: "2", label: "Client2", x: 400, y: 325, status: "idle" },
    { id: "3", label: "Client3", x: 100, y: 325, status: "idle" },
  ]);
  const [connections, setConnections] = useState<Connection[]>([
    { from: "1", to: "0", isActive: false },
    { from: "2", to: "0", isActive: false },
    { from: "3", to: "0", isActive: false },
  ]);
  const getNodeColor = (status: NodeStatus) => {
    switch (status) {
      case "running":
        return "fill-green-500"
      case "warning":
        return "fill-yellow-500"
      case "error":
        return "fill-red-500"
      case "idle" :
      case "stopped":
        return "fill-gray-300"
    }
  }

//触发训练脚本
  const handleTrainClick = async () => {
    try {
      await axios.post('http://127.0.0.1:5000/board/train');
      setTrainingStarted(true);
      console.info("触发训练脚本")
    } catch (error) {
      console.error('Error starting training:', error);
    }
  }

// 更新节点状态
const updateNodeStatus = useCallback((newStatus: NodeStatus) => {
  setNodes(prevNodes =>
      prevNodes.map(nodes => ({
          ...nodes,
          status: newStatus,
      }))
  );
},[]);
// 更新连接状态
const updateConnectionStatus = useCallback((isActive: boolean) => {
  setConnections(prevConnections =>
      prevConnections.map(connection => ({
          ...connection,
          isActive,
      }))
  );
},[]);

  //监听训练状态函数
  useEffect(() => {
    let eventSource: EventSource | null = null;
    let firstUpdated = false;
    let warningFlag = 0;
    if (trainingStarted) {
        eventSource = new EventSource('http://127.0.0.1:5000/board/train-status');
        console.info("sse ok!")
        eventSource.onmessage = (event) => {
            console.info("sse event ok!")
            const data = JSON.parse(event.data);
            console.log('Received data:', data);
            
            if (data.status === 'running') {
              //节点运行
              if (data.status === 'running' && !firstUpdated) {
                updateNodeStatus('running');
                updateConnectionStatus(true);
                firstUpdated = true;
              }

              //初步检测
              if('attacker' in data && data.attacker !== null){
                // 根据攻击者的 ID 更新特定节点状态为 warning
                setNodes(prevNodes => 
                  prevNodes.map(node => {
                    if (node.id === data.attacker) {
                      if (warningFlag < 5) {
                          warningFlag += 1;
                      }
                      return { ...node, status: 'warning' };
                    }
                    return node;
                  })
                );
              }

              //确认攻击者
              if('attacker' in data && data.attacker !== null && warningFlag === 5){
                setNodes(prevNodes => 
                  prevNodes.map(node => {
                    if (node.id === data.attacker && node.status === 'warning') {
                      // 如果状态仍为 warning，则更改为 error
                      return { ...node, status: 'error' };
                    }
                    return node;
                  })
                );
                // 断开与 arbiter 相连的特定节点的连接
                setConnections(prevConnections =>
                  prevConnections.map(connection => {
                    if (connection.from === data.attacker && connection.to === '0') {
                      return { ...connection, isActive: false };
                    }
                    return connection;
                  })
                );
              };
                console.info(warningFlag)
            } else if (data.status === 'stopped') {
                updateNodeStatus('stopped');
                updateConnectionStatus(false);
                setTrainingStarted(false);
                console.info(2222)
            }else {
                console.info(3333)
            }
        };
        eventSource.onerror = (error) => {
          if (eventSource && eventSource.readyState !== EventSource.CLOSED) {
            console.error("SSE connection error:", error);
            eventSource.close();
          }
        };
        return () => {
            if(eventSource) eventSource.close();
        };
    }
}, [trainingStarted, updateNodeStatus, updateConnectionStatus]);




  return (
    <div className="min-h-screen bg-background p-6">
      <header className="mb-6 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <img src="/immunity.png" alt="Immuniti Logo" className="h-8 w-auto" />
        </div>
        <div className="space-x-4">
          <Button
            variant="outline"
            className="rounded-full w-32 h-12 font-semibold hover:bg-primary hover:text-primary-foreground"
            onClick={handleTrainClick}
          >
            TRAIN
          </Button>
          <Button
            variant="outline"
            className="rounded-full w-32 h-12 font-semibold hover:bg-destructive hover:text-destructive-foreground"
            onClick={() => window.location.href = 'http://localhost:3000'}
          >
            <Power className="mr-2 h-4 w-4" />
            EXIT
          </Button>
        </div>
      </header>
      
      <div className="grid gap-6 md:grid-cols-2">
        <div className="space-y-2">
          <h2 className="text-lg font-medium">Party status</h2>
          <Card className="border-2 p-4">
            <div className="relative w-full" style={{ paddingTop: "100%" }}>
              <svg
                className="absolute inset-0 w-full h-full"
                viewBox="50 50 400 300"
                style={{ transform: "rotate(-90deg)" }}
              >
                {/* 画边 */}
                {connections.map((conn, i) => {
                  const fromNode = nodes.find(n => n.id === conn.from)!
                  const toNode = nodes.find(n => n.id === conn.to)!
                  return (
                    <line
                      key={i}
                      x1={fromNode.x}
                      y1={fromNode.y}
                      x2={toNode.x}
                      y2={toNode.y}
                      className={`stroke-2 transition-colors duration-300 ${
                        conn.isActive ? "stroke-gray-400" : "stroke-gray-200 stroke-dashed"
                      }`}
                      style={{ cursor: "pointer" }} 
                    />
                  )
                })}
                {/* 画点 */}
                {nodes.map((node) => (
                  <g
                    key={node.id}
                    transform={`translate(${node.x},${node.y}) rotate(90)`}
                    style={{ cursor: "pointer" }} 
                  >
                    <circle
                      r="30"
                      className={`${getNodeColor(node.status)} transition-colors duration-300 stroke-gray-400`}
                    />
                    <text
                      className="text-sm font-medium fill-gray-700"
                      textAnchor="middle"
                      dominantBaseline="middle"
                    >
                      {node.label}
                    </text>
                  </g>
                ))}
              </svg>
            </div>
          </Card>
        </div>

        <div className="space-y-2">
            <h2 className="text-lg font-medium">INFO</h2>
            {/* <Card className="h-[300px] border-2">
            <h1>Training Status: {trainingStarted}</h1>
                {trainingStarted === true && <p>Training in progress...</p>}
                {trainingStarted === false && <p>Training completed ~</p>}
              <h2>Node Status</h2>
              <ul>
                  {nodes.map(node => (
                      <li key={node.id}>
                          {node.label}: {node.status}
                      </li>
                  ))}
              </ul>
              <h2>Connection Status</h2>
              <ul>
                  {connections.map((connection, index) => (
                      <li key={index}>
                          {connection.from} → {connection.to}: {connection.isActive ? 'Active' : 'Inactive'}
                      </li>
                  ))}
              </ul>
            </Card> */}
                <InfoPanel 
                  trainingStarted={trainingStarted}
                  nodes={nodes}
                  connections={connections}
                />
          </div>
      </div>
      
    </div>
  )
}