"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, Download, BarChart3, Users, TrendingUp, Sparkles, Zap, Target, Brain } from "lucide-react"
import { FileUpload } from "@/components/file-upload"
import { PredictionResults } from "@/components/prediction-results"
import { ModelMetrics } from "@/components/model-metrics"
import { TrainingProgress } from "@/components/training-progress"

interface PredictionData {
  customerID: string
  churn_probability: number
  churn_prediction: number
  risk_level: string
}

interface ModelPerformance {
  auc_roc: number
  accuracy: number
  precision: number
  recall: number
  f1_score: number
  feature_importance: Array<{ feature: string; importance: number }>
}

export default function ChurnPredictionDashboard() {
  const [predictions, setPredictions] = useState<PredictionData[]>([])
  const [modelMetrics, setModelMetrics] = useState<ModelPerformance | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isTraining, setIsTraining] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [modelTrained, setModelTrained] = useState(false)

  useEffect(() => {
    // Check if model is already trained
    checkModelStatus()
  }, [])

  const checkModelStatus = async () => {
    try {
      const response = await fetch("/api/model-status")
      const data = await response.json()
      setModelTrained(data.trained)
      if (data.metrics) {
        setModelMetrics(data.metrics)
      }
    } catch (err) {
      console.error("Failed to check model status:", err)
    }
  }

  const handleTrainModel = async () => {
    setIsTraining(true)
    setError(null)

    try {
      const response = await fetch("/api/train", {
        method: "POST",
      })

      if (!response.ok) {
        throw new Error("Failed to train model")
      }

      const data = await response.json()
      setModelMetrics(data.metrics)
      setModelTrained(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Training failed")
    } finally {
      setIsTraining(false)
    }
  }

  const handleFileUpload = async (file: File) => {
    if (!modelTrained) {
      setError("Please train the model first before making predictions")
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append("file", file)

      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Failed to process predictions")
      }

      const data = await response.json()
      setPredictions(data.predictions)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setIsLoading(false)
    }
  }

  const downloadPredictions = () => {
    if (predictions.length === 0) return

    const csvContent = [
      "customerID,churn_probability,churn_prediction,risk_level",
      ...predictions.map((p) => `${p.customerID},${p.churn_probability},${p.churn_prediction},${p.risk_level}`),
    ].join("\n")

    const blob = new Blob([csvContent], { type: "text/csv" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "churn_predictions.csv"
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const churnCount = predictions.filter((p) => p.churn_prediction === 1).length
  const retainCount = predictions.length - churnCount
  const avgChurnProb =
    predictions.length > 0 ? predictions.reduce((sum, p) => sum + p.churn_probability, 0) / predictions.length : 0

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 text-white">
      {/* Animated background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500/10 rounded-full blur-3xl animate-pulse" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl animate-pulse delay-1000" />
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl animate-pulse delay-500" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto p-6 space-y-8">
        {/* Header */}
        <div className="text-center space-y-6 py-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-2xl shadow-2xl">
              <Brain className="h-8 w-8 text-white" />
            </div>
            <h1 className="text-5xl font-bold bg-gradient-to-r from-white via-gray-200 to-gray-400 bg-clip-text text-transparent">
             Fin-predicter
            </h1>
          </div>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Advanced machine learning platform for predicting customer churn with enterprise-grade accuracy and insights
          </p>
          <div className="flex items-center justify-center gap-2 text-sm text-gray-400">
            <Sparkles className="h-4 w-4 text-yellow-400" />
            <span>Powered by Advanced ML Algorithms</span>
          </div>
        </div>

        {/* Model Training Section */}
        {!modelTrained && (
          <Card className="bg-gradient-to-r from-gray-900/90 to-black/90 border-gray-700/50 backdrop-blur-xl shadow-2xl">
            <CardHeader>
              <CardTitle className="flex items-center gap-3 text-white">
                <div className="p-2 bg-gradient-to-r from-orange-500 to-red-500 rounded-lg">
                  <Zap className="h-5 w-5" />
                </div>
                Train Your Model
              </CardTitle>
              <CardDescription className="text-gray-300">
                Initialize the AI model with your customer data for accurate predictions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="p-6 bg-gradient-to-r from-gray-800/50 to-gray-900/50 rounded-xl border border-gray-700/30">
                  <h4 className="font-semibold text-white mb-2">Ready to train with your dataset</h4>
                  <p className="text-gray-400 text-sm mb-4">
                    The model will be trained on your telco customer data with advanced feature engineering and
                    optimization.
                  </p>
                  <Button
                    onClick={handleTrainModel}
                    disabled={isTraining}
                    className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-semibold px-6 py-3 rounded-xl shadow-lg transition-all duration-300 transform hover:scale-105"
                  >
                    {isTraining ? (
                      <div className="flex items-center gap-2">
                        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        Training Model...
                      </div>
                    ) : (
                      <div className="flex items-center gap-2">
                        <Brain className="h-4 w-4" />
                        Start Training
                      </div>
                    )}
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Training Progress */}
        {isTraining && <TrainingProgress />}

        {/* Quick Stats */}
        {predictions.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <Card className="bg-gradient-to-br from-gray-900/90 to-black/90 border-gray-700/50 backdrop-blur-xl shadow-2xl hover:shadow-purple-500/10 transition-all duration-300">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-gray-300">Total Customers</CardTitle>
                <div className="p-2 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg">
                  <Users className="h-4 w-4 text-white" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-white">{predictions.length.toLocaleString()}</div>
                <div className="h-1 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full mt-2" />
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-gray-900/90 to-black/90 border-gray-700/50 backdrop-blur-xl shadow-2xl hover:shadow-red-500/10 transition-all duration-300">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-gray-300">High Risk</CardTitle>
                <div className="p-2 bg-gradient-to-r from-red-500 to-pink-500 rounded-lg">
                  <TrendingUp className="h-4 w-4 text-white" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-red-400">{churnCount.toLocaleString()}</div>
                <p className="text-xs text-gray-400 mt-1">
                  {((churnCount / predictions.length) * 100).toFixed(1)}% of total
                </p>
                <div className="h-1 bg-gradient-to-r from-red-500 to-pink-500 rounded-full mt-2" />
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-gray-900/90 to-black/90 border-gray-700/50 backdrop-blur-xl shadow-2xl hover:shadow-green-500/10 transition-all duration-300">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-gray-300">Safe Customers</CardTitle>
                <div className="p-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg">
                  <BarChart3 className="h-4 w-4 text-white" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-green-400">{retainCount.toLocaleString()}</div>
                <p className="text-xs text-gray-400 mt-1">
                  {((retainCount / predictions.length) * 100).toFixed(1)}% retention rate
                </p>
                <div className="h-1 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full mt-2" />
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-gray-900/90 to-black/90 border-gray-700/50 backdrop-blur-xl shadow-2xl hover:shadow-yellow-500/10 transition-all duration-300">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-gray-300">Avg Risk Score</CardTitle>
                <div className="p-2 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-lg">
                  <Target className="h-4 w-4 text-white" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-yellow-400">{(avgChurnProb * 100).toFixed(1)}%</div>
                <div className="h-1 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-full mt-2" />
              </CardContent>
            </Card>
          </div>
        )}

        {/* File Upload */}
        {modelTrained && (
          <Card className="bg-gradient-to-br from-gray-900/90 to-black/90 border-gray-700/50 backdrop-blur-xl shadow-2xl">
            <CardHeader>
              <CardTitle className="flex items-center gap-3 text-white">
                <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg">
                  <Upload className="h-5 w-5" />
                </div>
                Upload Customer Data
              </CardTitle>
              <CardDescription className="text-gray-300">
                Upload your customer CSV file to generate AI-powered churn predictions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <FileUpload onFileUpload={handleFileUpload} isLoading={isLoading} />
              {error && (
                <div className="mt-4 p-4 bg-gradient-to-r from-red-900/50 to-red-800/50 border border-red-500/30 rounded-xl backdrop-blur-sm">
                  <p className="text-red-300 font-medium">{error}</p>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Model Performance Metrics */}
        {modelMetrics && <ModelMetrics metrics={modelMetrics} />}

        {/* Results */}
        {predictions.length > 0 && (
          <div className="space-y-8">
            <div className="flex justify-between items-center">
              <h2 className="text-3xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
                AI Prediction Results
              </h2>
              <Button
                onClick={downloadPredictions}
                className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white font-semibold px-6 py-3 rounded-xl shadow-lg transition-all duration-300 transform hover:scale-105"
              >
                <Download className="h-4 w-4 mr-2" />
                Export Results
              </Button>
            </div>

            <PredictionResults predictions={predictions} />
          </div>
        )}
      </div>
    </div>
  )
}
