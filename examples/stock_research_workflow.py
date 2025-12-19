#!/usr/bin/env python3
"""
Stock Research Workflow Example

Usage:
    # Using DeepSeek (default)
    DEEPSEEK_API_KEY=xxx python examples/stock_research_workflow.py 601288

    # Using environment variable
    export DEEPSEEK_API_KEY=xxx
    python examples/stock_research_workflow.py 00700

    # A-shares, HK, US markets all supported
    python examples/stock_research_workflow.py 600519   # A-shares: Kweichow Moutai
    python examples/stock_research_workflow.py 00700    # HK: Tencent
    python examples/stock_research_workflow.py AAPL     # US: Apple
"""

import asyncio
import sys
import os
import logging

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentkit.workflows.stock_research import research_stock, StockResearchWorkflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    ticker = sys.argv[1]

    # Check for API key - support both DEEPSEEK_API_KEY and ANTHROPIC_API_KEY
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    api_base = os.getenv("DEEPSEEK_API_BASE") or os.getenv("ANTHROPIC_API_BASE") or "https://api.deepseek.com"

    if not api_key:
        print("Error: Please set DEEPSEEK_API_KEY or ANTHROPIC_API_KEY in .env file or environment")
        print("\nExample .env:")
        print("  DEEPSEEK_API_KEY=sk-xxx")
        print("  # OR")
        print("  ANTHROPIC_API_KEY=sk-xxx")
        print("  ANTHROPIC_API_BASE=https://api.deepseek.com")
        sys.exit(1)

    print(f"API Base: {api_base}")

    # Detect market for display
    market, clean_ticker = StockResearchWorkflow.detect_market(ticker)
    market_names = {"A": "A股", "HK": "港股", "US": "美股"}

    print(f"\n{'='*60}")
    print(f"Stock Research Workflow")
    print(f"{'='*60}")
    print(f"Ticker: {clean_ticker}")
    print(f"Market: {market_names.get(market, market)}")
    print(f"Model:  deepseek-chat")
    print(f"{'='*60}\n")

    try:
        result = await research_stock(
            ticker=ticker,
            output_dir="./stock-research",
            api_key=api_key,
            api_base=api_base,
            model="deepseek-chat"
        )

        print(f"\n{'='*60}")
        print(f"Result: {result}")
        print(f"{'='*60}")

        if result.success:
            print(f"\n✓ Report saved to: {result.output_path}")

            # Show quality check results
            quality = result.data.get("quality_check_result", {})
            if quality:
                print(f"\nQuality Check:")
                print(f"  - File exists: {'✓' if quality.get('file_exists') else '✗'}")
                print(f"  - No placeholders: {'✓' if quality.get('no_placeholders') else '✗'}")
                print(f"  - Has rating: {'✓' if quality.get('has_rating') else '✗'}")
                print(f"  - Has prices: {'✓' if quality.get('has_prices') else '✗'}")
        else:
            print(f"\n✗ Workflow failed")
            for error in result.errors:
                print(f"  - {error}")
            sys.exit(1)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
