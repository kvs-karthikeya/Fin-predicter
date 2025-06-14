"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts"
import { Search, AlertTriangle, CheckCircle, TrendingUp, Shield, Zap } from "lucide-react"

interface PredictionData {
  customerID: string
  churn_probability: number
  churn_prediction: number
  risk_level: string
}

interface PredictionResultsProps {
  predictions: PredictionData[]
}

export function PredictionResults({ predictions }: PredictionResultsProps) {
  const [searchTerm, setSearchTerm] = useState("")
  const [churnThreshold, setChurnThreshold] = useState(0.5)

  // Filter predictions based on search term
  const filteredPredictions = predictions.filter((p) => p.customerID.toLowerCase().includes(searchTerm.toLowerCase()))

  // Sort by churn probability (highest first)
  const sortedPredictions = [...filteredPredictions].sort((a, b) => b.churn_probability - a.churn_probability)

  // Top 10 highest risk customers
  const topRiskCustomers = sortedPredictions.slice(0, 10)

  // Prepare data for histogram
  const histogramData = []
  const bins = 10
  for (let i = 0; i < bins; i++) {
    const binStart = i / bins
    const binEnd = (i + 1) / bins
    const count = predictions.filter((p) => p.churn_probability >= binStart && p.churn_probability < binEnd).length
    histogramData.push({
      range: `${(binStart * 100).toFixed(0)}-${(binEnd * 100).toFixed(0)}%`,
      count,
      binStart,
    })
  }

  // Prepare data for pie chart
  const churnCount = predictions.filter((p) => p.churn_probability >= churnThreshold).length
  const retainCount = predictions.length - churnCount
  const pieData = [
    { name: "High Risk", value: churnCount, color: "#ef4444" },
    { name: "Safe", value: retainCount, color: "#22c55e" },
  ]

  const getRiskIcon = (riskLevel: string) => {
    switch (riskLevel) {
      case "Critical":
        return <AlertTriangle className="h-4 w-4 text-red-400" />
      case "High":
        return <TrendingUp className="h-4 w-4 text-orange-400" />
      case "Medium":
        return <Zap className="h-4 w-4 text-yellow-400" />
      default:
        return <Shield className="h-4 w-4 text-green-400" />
    }
  }

  const getRiskBadgeVariant = (riskLevel: string) => {
    switch (riskLevel) {
      case "Critical":
        return "destructive"
      case "High":
        return "secondary"
      case "Medium":
        return "outline"
      default:
        return "default"
    }
  }

  return (
    <div className="space-y-8">
      {/* Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Churn Probability Distribution */}
        <Card className="bg-gradient-to-br from-gray-900/90 to-black/90 border-gray-700/50 backdrop-blur-xl shadow-2xl">
          <CardHeader>
            <CardTitle className="flex items-center gap-3 text-white">
              <div className="p-2 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg">
                <BarChart className="h-5 w-5" />
              </div>
              Risk Distribution Analysis
            </CardTitle>
            <CardDescription className="text-gray-300">
              Distribution of churn probabilities across your customer base
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={histogramData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="range" stroke="#9CA3AF" fontSize={12} />
                <YAxis stroke="#9CA3AF" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#1F2937",
                    border: "1px solid #374151",
                    borderRadius: "12px",
                    color: "#F9FAFB",
                  }}
                />
                <Bar dataKey="count" fill="url(#blueGradient)" radius={[4, 4, 0, 0]} />
                <defs>
                  <linearGradient id="blueGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#3B82F6" />
                    <stop offset="100%" stopColor="#1E40AF" />
                  </linearGradient>
                </defs>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Churn vs Retain Pie Chart */}
        <Card className="bg-gradient-to-br from-gray-900/90 to-black/90 border-gray-700/50 backdrop-blur-xl shadow-2xl">
          <CardHeader>
            <CardTitle className="flex items-center gap-3 text-white">
              <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg">
                <PieChart className="h-5 w-5" />
              </div>
              Customer Segmentation
            </CardTitle>
            <CardDescription className="text-gray-300">
              Risk-based customer categorization with adjustable threshold
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              <div className="flex items-center gap-4">
                <label htmlFor="threshold" className="text-sm font-medium text-gray-300 whitespace-nowrap">
                  Risk Threshold:
                </label>
                <div className="flex-1">
                  <Input
                    id="threshold"
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={churnThreshold}
                    onChange={(e) => setChurnThreshold(Number.parseFloat(e.target.value))}
                    className="w-full bg-gray-700 border-gray-600"
                  />
                  <div className="text-center mt-1 text-sm text-gray-400">{(churnThreshold * 100).toFixed(0)}%</div>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1F2937",
                      border: "1px solid #374151",
                      borderRadius: "12px",
                      color: "#F9FAFB",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Top Risk Customers Table */}
      <Card className="bg-gradient-to-br from-gray-900/90 to-black/90 border-gray-700/50 backdrop-blur-xl shadow-2xl">
        <CardHeader>
          <CardTitle className="flex items-center gap-3 text-white">
            <div className="p-2 bg-gradient-to-r from-red-500 to-pink-500 rounded-lg">
              <AlertTriangle className="h-5 w-5" />
            </div>
            Critical Risk Customers
          </CardTitle>
          <CardDescription className="text-gray-300">
            Top 10 customers with highest churn probability requiring immediate attention
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Search by customer ID..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 bg-gray-800/50 border-gray-600 text-white placeholder-gray-400 focus:border-purple-500 focus:ring-purple-500/20"
              />
            </div>

            <div className="rounded-xl overflow-hidden border border-gray-700/30">
              <Table>
                <TableHeader>
                  <TableRow className="bg-gradient-to-r from-gray-800/50 to-gray-900/50 border-gray-700/30">
                    <TableHead className="text-gray-300 font-semibold">Rank</TableHead>
                    <TableHead className="text-gray-300 font-semibold">Customer ID</TableHead>
                    <TableHead className="text-gray-300 font-semibold">Risk Score</TableHead>
                    <TableHead className="text-gray-300 font-semibold">Risk Level</TableHead>
                    <TableHead className="text-gray-300 font-semibold">Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {topRiskCustomers.map((customer, index) => (
                    <TableRow
                      key={customer.customerID}
                      className="border-gray-700/30 hover:bg-gray-800/30 transition-colors"
                    >
                      <TableCell className="font-medium text-white">
                        <div className="flex items-center gap-2">
                          <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white text-sm font-bold">
                            {index + 1}
                          </div>
                        </div>
                      </TableCell>
                      <TableCell className="text-gray-300 font-mono">{customer.customerID}</TableCell>
                      <TableCell>
                        <div className="space-y-2">
                          <div className="flex items-center gap-3">
                            <div className="flex-1 bg-gray-700 rounded-full h-2 overflow-hidden">
                              <div
                                className="bg-gradient-to-r from-red-500 to-pink-500 h-2 rounded-full transition-all duration-500"
                                style={{ width: `${customer.churn_probability * 100}%` }}
                              />
                            </div>
                            <span className="text-sm font-bold text-white min-w-[50px]">
                              {(customer.churn_probability * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant={getRiskBadgeVariant(customer.risk_level)}
                          className="flex items-center gap-1 w-fit"
                        >
                          {getRiskIcon(customer.risk_level)}
                          {customer.risk_level}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        {customer.churn_prediction === 1 ? (
                          <div className="flex items-center gap-2 text-red-400">
                            <AlertTriangle className="h-4 w-4" />
                            <span className="text-sm font-medium">At Risk</span>
                          </div>
                        ) : (
                          <div className="flex items-center gap-2 text-green-400">
                            <CheckCircle className="h-4 w-4" />
                            <span className="text-sm font-medium">Safe</span>
                          </div>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
