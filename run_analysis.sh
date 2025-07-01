#!/bin/bash
# Quick analysis runner script

set -e

echo "🔍 Starting paper comparison analysis..."

# Check if required CSV files exist
if [[ ! -f "papers.csv" ]]; then
    echo "❌ Error: papers.csv not found"
    exit 1
fi

if [[ ! -f "op_papers.csv" ]]; then
    echo "❌ Error: op_papers.csv not found"
    exit 1
fi

# Run the analysis
echo "📊 Running analysis..."
python paper_comparison.py

# Check if report was generated
if [[ -f "paper_comparison_report.html" ]]; then
    echo "✅ Analysis complete!"
    echo "📄 Report generated: paper_comparison_report.html"
    
    # Try to open the report in the default browser (macOS)
    if command -v open >/dev/null 2>&1; then
        echo "🌐 Opening report in browser..."
        open paper_comparison_report.html
    fi
else
    echo "❌ Error: Report generation failed"
    exit 1
fi