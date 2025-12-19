"""Stock Research Agent - Deep equity research with fund manager perspective"""
import asyncio
import re
from pathlib import Path
from typing import Any, Optional
from agentkit.core.agent import Agent
from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock, ToolResultBlock


class StockResearchAgent(Agent):
    """
    Agent for comprehensive stock research across A/HK/US markets

    Features:
    - Market detection based on ticker format
    - Data fetching via external script
    - Report generation with AI analysis
    - Quality validation
    """

    SYSTEM_PROMPT = """You are a professional equity research analyst with fund manager perspective.

Your task is to conduct deep stock research following this workflow:

## Market Detection
- 6-digit numeric (600519) → A-shares, path: A/
- 5-digit numeric (00700) → HK, path: HK/
- Alphabetic 1-5 chars (AAPL) → US, path: US/

## Analysis Framework
When analyzing stocks, cover these areas:
1. 摘要速览: Core investment thesis, ROE, FCFF status, risk level
2. 公司概况: Business description, history, management
3. 财务分析: YoY comparison, industry benchmarks, quality assessment
4. FCFF诊断: Value trap risk based on free cash flow
5. 业务结构: Revenue breakdown, growth drivers, moat type
6. 竞争格局: Market position, competitors, industry trends
7. 风险因素: Risk matrix with probability/impact ratings
8. 估值分析: Historical/peer comparison, fair value range
9. 技术结论: Entry/stop/target prices based on TA
10. 基金经理视角: Position sizing, risk exposure, liquidity
11. 投资观点: 3 bull points, 3 bear points, final rating
12. 催化剂: Future dates only

## Tools Available
- Use Bash to run data scripts
- Use Read/Write/Edit for file operations
- Use WebSearch for latest market data

Be thorough, data-driven, and objective in your analysis."""

    def __init__(
        self,
        output_dir: str = "./stock-research",
        **kwargs
    ):
        super().__init__(
            system_prompt=self.SYSTEM_PROMPT,
            tools={"type": "preset", "preset": "claude_code"},
            permission_mode="bypassPermissions",  # Allow autonomous operation
            **kwargs
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def detect_market(ticker: str) -> tuple[str, str]:
        """Detect market from ticker format"""
        ticker = ticker.strip().upper()

        if re.match(r'^\d{6}$', ticker):
            return "A", ticker
        elif re.match(r'^\d{5}$', ticker):
            return "HK", ticker
        elif re.match(r'^[A-Z]{1,5}$', ticker):
            return "US", ticker
        else:
            raise ValueError(f"Unknown ticker format: {ticker}")

    async def process_response(self, message: AssistantMessage) -> dict[str, Any]:
        """Process and extract structured response"""
        result = {
            "text": [],
            "tool_calls": [],
            "tool_results": []
        }

        for block in message.content:
            if isinstance(block, TextBlock):
                result["text"].append(block.text)
            elif isinstance(block, ToolUseBlock):
                result["tool_calls"].append({
                    "name": block.name,
                    "input": block.input
                })
            elif isinstance(block, ToolResultBlock):
                result["tool_results"].append({
                    "tool_use_id": block.tool_use_id,
                    "content": block.content
                })

        return result

    async def research(self, ticker: str, options: Optional[dict] = None) -> str:
        """
        Conduct stock research for given ticker

        Args:
            ticker: Stock ticker (e.g., 600519, 00700, AAPL)
            options: Optional research options (--update, --skip-ta, etc.)

        Returns:
            Path to generated report
        """
        options = options or {}
        market, clean_ticker = self.detect_market(ticker)

        # Build the research prompt
        prompt = f"""Research the stock: {ticker}

Market: {market}
Output directory: {self.output_dir}/{market}/

Steps to follow:
1. Search for latest company information and financial data
2. Analyze the business model and competitive position
3. Evaluate financial health (ROE, FCFF, debt levels)
4. Assess valuation relative to peers and history
5. Identify key risks and catalysts
6. Provide investment recommendation with entry/stop/target

Create a comprehensive research report at: {self.output_dir}/{market}/{clean_ticker}.md

Use WebSearch to get the latest data. Be thorough and objective."""

        if options.get("update"):
            prompt += "\n\nThis is an UPDATE - focus on recent changes and news."
        if options.get("section"):
            prompt += f"\n\nFocus specifically on section: {options['section']}"

        # Run the research
        async with self:
            result = await self.chat(prompt)

        report_path = self.output_dir / market / f"{clean_ticker}.md"
        return str(report_path)


async def main():
    """Demo: Research a stock"""
    import sys

    # Parse command line args
    if len(sys.argv) < 2:
        print("Usage: python stock_research_agent.py <ticker> [--update] [--debug]")
        print("Examples:")
        print("  python stock_research_agent.py AAPL")
        print("  python stock_research_agent.py 600519 --update")
        print("  python stock_research_agent.py 00700 --debug")
        sys.exit(1)

    ticker = sys.argv[1]
    options = {"update": "--update" in sys.argv}
    debug = "--debug" in sys.argv

    print(f"Starting research for {ticker}...")

    agent = StockResearchAgent(output_dir="./stock-research")

    try:
        # For debug mode, show raw response
        if debug:
            market, clean_ticker = agent.detect_market(ticker)
            prompt = f"""Research the stock: {ticker}

Market: {market}
Output directory: {agent.output_dir}/{market}/

Search for the latest company information and create a brief research summary.
Write the report to: {agent.output_dir}/{market}/{clean_ticker}.md"""

            async with agent:
                response = await agent.chat(prompt)
                print("\n=== Agent Response ===")
                print(response[:2000] if len(response) > 2000 else response)
                print("=== End Response ===\n")
        else:
            report_path = await agent.research(ticker, options)
            print(f"\nResearch complete!")
            print(f"Report saved to: {report_path}")

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        if debug:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
