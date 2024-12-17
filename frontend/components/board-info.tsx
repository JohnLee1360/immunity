'use client'

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Activity, Circle, ArrowRight } from 'lucide-react'
import { motion } from "framer-motion"

interface InfoPanelProps {
  trainingStarted: boolean
  nodes: Array<{
    id: string
    label: string
    status: string
  }>
  connections: Array<{
    from: string
    to: string
    isActive: boolean
  }>
}

export default function Component({ 
  trainingStarted = false,
  nodes = [],
  connections = []
}: InfoPanelProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'bg-green-500'
      case 'warning':
        return 'bg-yellow-500'
      case 'error':
        return 'bg-red-500'
      default:
        return 'bg-gray-300'
    }
  }

  const getNodeLabel = (id: string) => {
    switch (id) {
      case '0':
        return 'Arbiter'
      case '1':
        return 'Client 1'
      case '2':
        return 'Client 2'
      case '3':
        return 'Client 3'
      default:
        return id
    }
  }

  return (
    <Card className="relative w-full" style={{ paddingTop: "100%" }}>
      <div className="absolute inset-0 p-6 overflow-auto">
        {/* Training Status Section */}
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-sm font-medium text-gray-500">Training Status</h3>
          <Badge 
            variant={trainingStarted ? "default" : "secondary"}
            className="rounded-full px-3"
          >
            <Activity className="w-3 h-3 mr-1" />
            {trainingStarted ? "In Progress" : "Completed"}
          </Badge>
        </div>

        {/* Node Status Section */}
        <div className="space-y-4 mb-6">
          <h3 className="text-sm font-medium text-gray-500">Node Status</h3>
          <div className="space-y-3">
            {nodes.map((node) => (
              <motion.div
                key={node.id}
                className="flex items-center justify-between p-3 rounded-lg bg-white/50 border border-gray-100"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <div className="flex items-center space-x-2">
                  <Circle className={`w-2 h-2 ${getStatusColor(node.status)}`} />
                  <span className="text-sm font-medium">{node.label}</span>
                </div>
                <Badge 
                  variant="outline" 
                  className="text-xs capitalize"
                >
                  {node.status}
                </Badge>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Connection Status Section */}
        <div className="space-y-4">
          <h3 className="text-sm font-medium text-gray-500">Connection Status</h3>
          <div className="space-y-3">
            {connections.map((connection, index) => (
              <motion.div
                key={index}
                className="flex items-center justify-between p-3 rounded-lg bg-white/50 border border-gray-100"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
              >
                <div className="flex items-center space-x-2">
                  <span className="text-sm">{getNodeLabel(connection.from)}</span>
                  <ArrowRight className="w-3 h-3 text-gray-400" />
                  <span className="text-sm">{getNodeLabel(connection.to)}</span>
                </div>
                <Badge 
                  variant={connection.isActive ? "default" : "secondary"}
                  className="text-xs"
                >
                  {connection.isActive ? "Active" : "Inactive"}
                </Badge>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </Card>
  )
}