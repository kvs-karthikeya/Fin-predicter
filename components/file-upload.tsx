"use client"

import { useCallback, useState } from "react"
import { useDropzone } from "react-dropzone"
import { Button } from "@/components/ui/button"
import { Upload, FileText, Loader2, Sparkles, CheckCircle } from "lucide-react"

interface FileUploadProps {
  onFileUpload: (file: File) => void
  isLoading: boolean
}

export function FileUpload({ onFileUpload, isLoading }: FileUploadProps) {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (file) {
      setUploadedFile(file)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/csv": [".csv"],
    },
    multiple: false,
  })

  const handleUpload = () => {
    if (uploadedFile) {
      onFileUpload(uploadedFile)
    }
  }

  return (
    <div className="space-y-6">
      <div
        {...getRootProps()}
        className={`relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300 ${
          isDragActive
            ? "border-purple-400 bg-gradient-to-r from-purple-500/10 to-pink-500/10 scale-105"
            : "border-gray-600 hover:border-gray-500 bg-gradient-to-r from-gray-800/30 to-gray-900/30"
        } backdrop-blur-sm`}
      >
        <input {...getInputProps()} />
        <div className="space-y-6">
          <div className="mx-auto w-20 h-20 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center shadow-2xl">
            <Upload className="h-10 w-10 text-white" />
          </div>
          {isDragActive ? (
            <div className="space-y-2">
              <p className="text-xl font-semibold text-purple-300">Drop your CSV file here</p>
              <div className="flex items-center justify-center gap-2">
                <Sparkles className="h-4 w-4 text-yellow-400 animate-pulse" />
                <span className="text-gray-400">AI is ready to process your data</span>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <p className="text-xl font-semibold text-white">Upload Customer Data</p>
              <p className="text-gray-400 max-w-md mx-auto">
                Drag and drop your CSV file here, or click to browse and select your customer dataset
              </p>
              <div className="flex items-center justify-center gap-4 text-sm text-gray-500">
                <span>• CSV format only</span>
                <span>• Up to 50MB</span>
                <span>• Instant processing</span>
              </div>
            </div>
          )}
        </div>

        {/* Animated border effect */}
        <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-purple-500/20 to-pink-500/20 opacity-0 hover:opacity-100 transition-opacity duration-300 pointer-events-none" />
      </div>

      {uploadedFile && (
        <div className="bg-gradient-to-r from-gray-800/50 to-gray-900/50 rounded-2xl p-6 border border-gray-700/30 backdrop-blur-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl">
                <FileText className="h-6 w-6 text-white" />
              </div>
              <div>
                <p className="font-semibold text-white text-lg">{uploadedFile.name}</p>
                <div className="flex items-center gap-4 text-sm text-gray-400">
                  <span>{(uploadedFile.size / 1024 / 1024).toFixed(2)} MB</span>
                  <div className="flex items-center gap-1">
                    <CheckCircle className="h-4 w-4 text-green-400" />
                    <span>Ready for processing</span>
                  </div>
                </div>
              </div>
            </div>
            <Button
              onClick={handleUpload}
              disabled={isLoading}
              className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-semibold px-8 py-3 rounded-xl shadow-lg transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <div className="flex items-center gap-3">
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span>Processing...</span>
                </div>
              ) : (
                <div className="flex items-center gap-3">
                  <Sparkles className="h-5 w-5" />
                  <span>Generate Predictions</span>
                </div>
              )}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
