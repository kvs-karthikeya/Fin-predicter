import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get("file") as File

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 })
    }

    // Read CSV file
    const text = await file.text()
    const lines = text.split("\n").filter((line) => line.trim())

    if (lines.length < 2) {
      return NextResponse.json({ error: "Invalid CSV format" }, { status: 400 })
    }

    // Parse CSV headers and data
    const headers = lines[0].split(",").map((h) => h.trim())
    const data = lines.slice(1).map((line) => {
      const values = line.split(",").map((v) => v.trim())
      const row: Record<string, string> = {}
      headers.forEach((header, index) => {
        row[header] = values[index] || ""
      })
      return row
    })

    // Generate realistic predictions based on customer features
    const predictions = data.map((row) => {
      const customerID = row.customerID || row.customer_id || `customer_${Math.random().toString(36).substr(2, 9)}`

      // Extract key features for prediction logic
      const tenure = Number.parseInt(row.tenure) || 0
      const monthlyCharges = Number.parseFloat(row.MonthlyCharges || row.monthly_charges) || 0
      const contract = row.Contract || row.contract || ""
      const paymentMethod = row.PaymentMethod || row.payment_method || ""
      const internetService = row.InternetService || row.internet_service || ""
      const techSupport = row.TechSupport || row.tech_support || ""

      // Realistic churn probability calculation based on telco patterns
      let churn_probability = 0.1 // Base probability

      // Tenure impact (newer customers more likely to churn)
      if (tenure < 6) churn_probability += 0.4
      else if (tenure < 12) churn_probability += 0.3
      else if (tenure < 24) churn_probability += 0.2
      else if (tenure < 36) churn_probability += 0.1

      // Contract type impact
      if (contract.toLowerCase().includes("month")) churn_probability += 0.3
      else if (contract.toLowerCase().includes("one")) churn_probability += 0.1

      // Payment method impact
      if (paymentMethod.toLowerCase().includes("electronic")) churn_probability += 0.2

      // Monthly charges impact
      if (monthlyCharges > 80) churn_probability += 0.2
      else if (monthlyCharges > 60) churn_probability += 0.1

      // Internet service impact
      if (internetService.toLowerCase().includes("fiber")) churn_probability += 0.1

      // Tech support impact
      if (techSupport.toLowerCase().includes("no")) churn_probability += 0.15

      // Add some randomness but keep it realistic
      churn_probability += (Math.random() - 0.5) * 0.2

      // Ensure probability is between 0 and 1
      churn_probability = Math.max(0.01, Math.min(0.99, churn_probability))

      // Determine risk level
      let risk_level = "Low"
      if (churn_probability >= 0.8) risk_level = "Critical"
      else if (churn_probability >= 0.6) risk_level = "High"
      else if (churn_probability >= 0.4) risk_level = "Medium"

      return {
        customerID,
        churn_probability: Number.parseFloat(churn_probability.toFixed(3)),
        churn_prediction: churn_probability >= 0.5 ? 1 : 0,
        risk_level,
      }
    })

    return NextResponse.json({
      predictions,
      total_customers: predictions.length,
      churn_count: predictions.filter((p) => p.churn_prediction === 1).length,
      high_risk_count: predictions.filter((p) => p.risk_level === "Critical" || p.risk_level === "High").length,
    })
  } catch (error) {
    console.error("Prediction error:", error)
    return NextResponse.json({ error: "Failed to process predictions" }, { status: 500 })
  }
}
