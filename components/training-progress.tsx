"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Brain, Zap, Target, TrendingUp } from "lucide-react"

export function TrainingProgress() {
  return (
    <Card className="bg-gradient-to-br from-gray-900/90 to-black/90 border-gray-700/50 backdrop-blur-xl shadow-2xl">
      <CardHeader>
        <CardTitle className="flex items-center gap-3 text-white">
          <div className="p-2 bg-gradient-to-r from-orange-500 to-red-500 rounded-lg">
            <Brain className="h-5 w-5 animate-pulse" />
          </div>
          Training AI Model
        </CardTitle>
        <CardDescription className="text-gray-300">
          Processing your data and optimizing the machine learning model
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4 text-blue-400" />
                <span className="text-sm text-gray-300">Data Processing</span>
              </div>
              <Progress value={100} className="h-2 bg-gray-700" />
              <span className="text-xs text-green-400">Complete</span>
            </div>

            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Target className="h-4 w-4 text-purple-400" />
                <span className="text-sm text-gray-300">Model Training</span>
              </div>
              <Progress value={75} className="h-2 bg-gray-700" />
              <span className="text-xs text-yellow-400">In Progress</span>
            </div>

            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-gray-400" />
                <span className="text-sm text-gray-300">Optimization</span>
              </div>
              <Progress value={0} className="h-2 bg-gray-700" />
              <span className="text-xs text-gray-400">Pending</span>
            </div>
          </div>

          <div className="p-4 bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-xl border border-blue-500/20">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
              <span className="text-blue-300 font-medium">Training Status</span>
            </div>
            <p className="text-sm text-gray-300">Analyzing customer patterns and building predictive features...</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
