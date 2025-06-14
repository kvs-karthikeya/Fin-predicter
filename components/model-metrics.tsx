"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { TrendingUp, Target, Zap, BarChart3, Brain, Award } from "lucide-react"

interface ModelPerformance {
  auc_roc: number
  accuracy: number
  precision: number
  recall: number
  f1_score: number
  feature_importance: Array<{ feature: string; importance: number }>
}

interface ModelMetricsProps {
  metrics: ModelPerformance
}

export function ModelMetrics({ metrics }: ModelMetricsProps) {
  const getScoreColor = (score: number) => {
    if (score >= 0.9) return "text-green-400"
    if (score >= 0.8) return "text-blue-400"
    if (score >= 0.7) return "text-yellow-400"
    return "text-red-400"
  }

  const getScoreBadge = (score: number) => {
    if (score >= 0.9) return { label: "Excellent", color: "bg-green-500" }
    if (score >= 0.8) return { label: "Good", color: "bg-blue-500" }
    if (score >= 0.7) return { label: "Fair", color: "bg-yellow-500" }
    return { label: "Poor", color: "bg-red-500" }
  }

  const getProgressColor = (score: number) => {
    if (score >= 0.9) return "bg-gradient-to-r from-green-500 to-emerald-500"
    if (score >= 0.8) return "bg-gradient-to-r from-blue-500 to-cyan-500"
    if (score >= 0.7) return "bg-gradient-to-r from-yellow-500 to-orange-500"
    return "bg-gradient-to-r from-red-500 to-pink-500"
  }

  return (
    <div className="space-y-8">
      <Card className="bg-gradient-to-br from-gray-900/90 to-black/90 border-gray-700/50 backdrop-blur-xl shadow-2xl">
        <CardHeader>
          <CardTitle className="flex items-center gap-3 text-white">
            <div className="p-2 bg-gradient-to-r from-purple-500 to-blue-500 rounded-lg">
              <Brain className="h-5 w-5" />
            </div>
            AI Model Performance
          </CardTitle>
          <CardDescription className="text-gray-300">
            Comprehensive evaluation metrics of your trained churn prediction model
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg">
                  <TrendingUp className="h-4 w-4 text-white" />
                </div>
                <span className="text-sm font-medium text-gray-300">AUC-ROC</span>
              </div>
              <div className={`text-3xl font-bold ${getScoreColor(metrics.auc_roc)}`}>{metrics.auc_roc.toFixed(3)}</div>
              <div className="space-y-2">
                <Progress value={metrics.auc_roc * 100} className="h-2 bg-gray-700" />
                <Badge className={`${getScoreBadge(metrics.auc_roc).color} text-white text-xs`}>
                  {getScoreBadge(metrics.auc_roc).label}
                </Badge>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg">
                  <Target className="h-4 w-4 text-white" />
                </div>
                <span className="text-sm font-medium text-gray-300">Accuracy</span>
              </div>
              <div className={`text-3xl font-bold ${getScoreColor(metrics.accuracy)}`}>
                {(metrics.accuracy * 100).toFixed(1)}%
              </div>
              <div className="space-y-2">
                <Progress value={metrics.accuracy * 100} className="h-2 bg-gray-700" />
                <Badge className={`${getScoreBadge(metrics.accuracy).color} text-white text-xs`}>
                  {getScoreBadge(metrics.accuracy).label}
                </Badge>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg">
                  <Zap className="h-4 w-4 text-white" />
                </div>
                <span className="text-sm font-medium text-gray-300">Precision</span>
              </div>
              <div className={`text-3xl font-bold ${getScoreColor(metrics.precision)}`}>
                {(metrics.precision * 100).toFixed(1)}%
              </div>
              <div className="space-y-2">
                <Progress value={metrics.precision * 100} className="h-2 bg-gray-700" />
                <Badge className={`${getScoreBadge(metrics.precision).color} text-white text-xs`}>
                  {getScoreBadge(metrics.precision).label}
                </Badge>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-gradient-to-r from-orange-500 to-red-500 rounded-lg">
                  <BarChart3 className="h-4 w-4 text-white" />
                </div>
                <span className="text-sm font-medium text-gray-300">Recall</span>
              </div>
              <div className={`text-3xl font-bold ${getScoreColor(metrics.recall)}`}>
                {(metrics.recall * 100).toFixed(1)}%
              </div>
              <div className="space-y-2">
                <Progress value={metrics.recall * 100} className="h-2 bg-gray-700" />
                <Badge className={`${getScoreBadge(metrics.recall).color} text-white text-xs`}>
                  {getScoreBadge(metrics.recall).label}
                </Badge>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-lg">
                  <Award className="h-4 w-4 text-white" />
                </div>
                <span className="text-sm font-medium text-gray-300">F1-Score</span>
              </div>
              <div className={`text-3xl font-bold ${getScoreColor(metrics.f1_score)}`}>
                {(metrics.f1_score * 100).toFixed(1)}%
              </div>
              <div className="space-y-2">
                <Progress value={metrics.f1_score * 100} className="h-2 bg-gray-700" />
                <Badge className={`${getScoreBadge(metrics.f1_score).color} text-white text-xs`}>
                  {getScoreBadge(metrics.f1_score).label}
                </Badge>
              </div>
            </div>
          </div>

          <div className="mt-8 p-6 bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-2xl border border-blue-500/20">
            <h4 className="font-semibold text-blue-300 mb-4 flex items-center gap-2">
              <Brain className="h-5 w-5" />
              AI Model Insights
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-blue-200">
                  <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                  <span>
                    AUC-ROC of {metrics.auc_roc.toFixed(3)} indicates{" "}
                    {getScoreBadge(metrics.auc_roc).label.toLowerCase()} discrimination
                  </span>
                </div>
                <div className="flex items-center gap-2 text-blue-200">
                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  <span>Model correctly identifies {(metrics.recall * 100).toFixed(1)}% of actual churners</span>
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-blue-200">
                  <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                  <span>{(metrics.precision * 100).toFixed(1)}% of predicted churners are actually at risk</span>
                </div>
                <div className="flex items-center gap-2 text-blue-200">
                  <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
                  <span>Overall accuracy of {(metrics.accuracy * 100).toFixed(1)}% across all predictions</span>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Feature Importance */}
      {metrics.feature_importance && metrics.feature_importance.length > 0 && (
        <Card className="bg-gradient-to-br from-gray-900/90 to-black/90 border-gray-700/50 backdrop-blur-xl shadow-2xl">
          <CardHeader>
            <CardTitle className="flex items-center gap-3 text-white">
              <div className="p-2 bg-gradient-to-r from-green-500 to-teal-500 rounded-lg">
                <BarChart3 className="h-5 w-5" />
              </div>
              Feature Importance Analysis
            </CardTitle>
            <CardDescription className="text-gray-300">
              Most influential factors in predicting customer churn
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {metrics.feature_importance.slice(0, 8).map((feature, index) => (
                <div key={feature.feature} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white text-sm font-bold">
                        {index + 1}
                      </div>
                      <span className="font-medium text-white capitalize">{feature.feature.replace(/_/g, " ")}</span>
                    </div>
                    <span className="text-sm font-bold text-gray-300">{(feature.importance * 100).toFixed(1)}%</span>
                  </div>
                  <div className="ml-11">
                    <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all duration-1000"
                        style={{
                          width: `${(feature.importance / Math.max(...metrics.feature_importance.map((f) => f.importance))) * 100}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
