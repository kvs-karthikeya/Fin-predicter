import { NextResponse } from "next/server"

export async function GET() {
  try {
    // In a real implementation, this would check if a trained model exists
    // For demo purposes, we'll return that no model is trained initially

    return NextResponse.json({
      trained: false,
      metrics: null,
      last_trained: null,
    })
  } catch (error) {
    console.error("Model status error:", error)
    return NextResponse.json({ error: "Failed to check model status" }, { status: 500 })
  }
}
