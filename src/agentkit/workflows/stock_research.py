"""
Stock Research Workflow

Implements the structured stock research workflow:
1. Fetch data using akshare
2. Generate report skeleton
3. Fill analysis using LLM
4. Write report file
5. Verify output
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import logging

from .base import Workflow, WorkflowResult

logger = logging.getLogger(__name__)


class StockResearchWorkflow(Workflow):
    """
    Stock research workflow with fund manager perspective.

    Supports A/HK/US markets with structured analysis.
    """

    # Equity research methodology (from skill.md)
    ANALYSIS_PROMPT = """你是一位专业的基金经理，正在进行股票深度研究。

## 分析框架

### 1. 自由现金流优先 (价值陷阱检测)
- FCFF = 经营活动现金流 - 资本支出
- 持续为正 = 价值创造者，可长期持有
- 波动正负 = 周期性，只做波段
- 持续为负 = ⚠️ 价值陷阱，避免长期持仓

### 2. 护城河评估
- 宽护城河: 多重优势来源，ROIC > 15% 持续10年+
- 窄护城河: 单一优势来源，ROIC > 10% 持续5年+
- 无护城河: 无可持续竞争优势

### 3. 风险矩阵
- 高概率+高影响: 仓位<5%，设置止损
- 高概率+低影响: 接受，密切监控
- 低概率+高影响: 对冲或减少敞口

## 你的任务

根据以下股票数据，填写分析内容。每个分析要具体、可操作：
- ❌ "考虑买入" → ✅ "在22-24区间买入，20止损"
- ❌ "可能有风险" → ✅ "原材料涨20%则毛利率下降3%"
- ❌ "估值合理" → ✅ "合理价28-32，当前25，上行12%"

## 股票数据

{stock_data}

## 需要填写的内容

请用中文填写以下分析（不需要格式标记，直接给出内容）：

1. **核心投资逻辑** (一句话)
2. **ROE水平** (百分比)
3. **近5年FCFF状态** (持续为正/波动/持续为负)
4. **价值陷阱风险** (低/中/高)
5. **建议操作** (建仓/加仓/持有/减仓/清仓)
6. **目标仓位** (占组合百分比)
7. **适合策略** (长期持有/波段操作/不建议)
8. **主要风险一句话**
9. **公司概况** (100-150字，主营业务、发展历程、管理层)
10. **行业分析** (行业地位、竞争格局、趋势)
11. **财务质量评估** (现金流、应收账款、存货、分红)
12. **FCFF诊断结论** (基于数据判断是否为价值陷阱)
13. **业务收入构成** (主要业务及占比)
14. **增长驱动因素** (3个主要增长引擎)
15. **护城河类型** (品牌/网络效应/成本优势/转换成本/无形资产)
16. **护城河强度** (宽/窄/无)
17. **竞争对手分析** (主要竞争者及市场份额)
18. **风险矩阵** (列出3-4个主要风险，标注概率和影响)
19. **估值评估** (PE/PB的历史对比、行业对比)
20. **合理价格区间**
21. **安全边际** (百分比)
22. **技术结论** (趋势、支撑阻力、操作建议)
23. **入场区间价格**
24. **止损价格**
25. **目标价格**
26. **看多理由** (3个核心论点)
27. **看空理由** (3个核心论点)
28. **投资评级** (强烈推荐/推荐/中性/回避/强烈回避)
29. **催化剂** (未来2-3个关键时间点和事件)

请以JSON格式返回，key使用英文:
{{
  "core_thesis": "...",
  "roe": "...",
  "fcff_status": "...",
  "value_trap_risk": "...",
  "action": "...",
  "position_size": "...",
  "strategy": "...",
  "key_risk": "...",
  "company_overview": "...",
  "industry_analysis": "...",
  "financial_quality": "...",
  "fcff_conclusion": "...",
  "revenue_breakdown": "...",
  "growth_drivers": "...",
  "moat_type": "...",
  "moat_width": "...",
  "competitors": "...",
  "risk_matrix": "...",
  "valuation_assessment": "...",
  "fair_value_range": "...",
  "margin_of_safety": "...",
  "technical_conclusion": "...",
  "entry_price": "...",
  "stop_loss": "...",
  "target_price": "...",
  "bull_case": ["...", "...", "..."],
  "bear_case": ["...", "...", "..."],
  "rating": "...",
  "catalysts": "..."
}}
"""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = "deepseek-chat"
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.api_base = api_base or os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        self.model = model
        super().__init__("stock_research", output_dir)

    def _setup_steps(self) -> None:
        """Define the 5-step workflow"""
        self.add_step(
            name="fetch_data",
            description="Fetch stock data using akshare",
            action=self._fetch_data,
            verify=lambda r, ctx: r is not None and "ticker" in r,
            required=True
        )

        self.add_step(
            name="generate_skeleton",
            description="Generate report skeleton with placeholders",
            action=self._generate_skeleton,
            verify=lambda r, ctx: r is not None and "{{CLAUDE_FILL" in r,
            required=True
        )

        self.add_step(
            name="fill_analysis",
            description="Fill analysis using LLM",
            action=self._fill_analysis,
            verify=lambda r, ctx: r is not None,
            required=True
        )

        self.add_step(
            name="write_report",
            description="Write completed report to file",
            action=self._write_report,
            verify=lambda r, ctx: r.exists() if isinstance(r, Path) else False,
            required=True
        )

        self.add_step(
            name="quality_check",
            description="Verify report quality",
            action=self._quality_check,
            verify=lambda r, ctx: r.get("passed", False),
            required=True
        )

    @staticmethod
    def detect_market(ticker: str) -> tuple[str, str]:
        """Detect market from ticker format"""
        ticker = ticker.upper().replace(".HK", "").replace(".SS", "").replace(".SZ", "")

        if ticker.isdigit():
            if len(ticker) == 6:
                return "A", ticker
            elif len(ticker) == 5:
                return "HK", ticker
        elif ticker.isalpha() and 1 <= len(ticker) <= 5:
            return "US", ticker

        # Default fallback based on prefix
        if ticker.startswith(("60", "00", "30", "68")):
            return "A", ticker

        return "A", ticker

    def _fetch_data(self, ctx: dict) -> dict:
        """Step 1: Fetch stock data using akshare"""
        import akshare as ak
        import pandas as pd

        ticker = ctx["ticker"]
        market, clean_ticker = self.detect_market(ticker)
        ctx["market"] = market
        ctx["clean_ticker"] = clean_ticker

        data = {
            "ticker": clean_ticker,
            "market": market,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "info": {},
            "financials": {},
            "valuation": {},
            "trading": {}
        }

        try:
            if market == "A":
                # Stock info
                try:
                    df = ak.stock_individual_info_em(symbol=clean_ticker)
                    info_map = dict(zip(df["item"], df["value"]))
                    data["info"] = {
                        "name": info_map.get("股票简称", clean_ticker),
                        "industry": info_map.get("行业", ""),
                        "market_cap": info_map.get("总市值", ""),
                        "float_cap": info_map.get("流通市值", ""),
                        "pe_ttm": info_map.get("市盈率(动态)", ""),
                        "pb": info_map.get("市净率", "")
                    }
                except Exception as e:
                    logger.warning(f"Could not fetch stock info: {e}")
                    data["info"]["name"] = clean_ticker

                # Financial data
                try:
                    df = ak.stock_financial_abstract_ths(symbol=clean_ticker, indicator="按报告期")
                    if not df.empty and "报告期" in df.columns:
                        df = df.sort_values("报告期", ascending=False)
                        latest = df.head(4)
                        data["financials"]["periods"] = latest["报告期"].tolist()

                        for key, col in [
                            ("revenue", "营业总收入"),
                            ("net_profit", "净利润"),
                            ("gross_margin", "毛利率"),
                            ("roe", "净资产收益率"),
                            ("debt_ratio", "资产负债率")
                        ]:
                            if col in latest.columns:
                                data["financials"][key] = latest[col].tolist()
                except Exception as e:
                    logger.warning(f"Could not fetch financial data: {e}")

                # Cash flow for FCFF
                try:
                    cf = ak.stock_financial_cash_ths(symbol=clean_ticker, indicator="按报告期")
                    if not cf.empty and "报告期" in cf.columns:
                        cf = cf.sort_values("报告期", ascending=False)
                        ocf_col = "经营活动产生的现金流量净额"
                        capex_col = "购建固定资产、无形资产和其他长期资产支付的现金"

                        if ocf_col in cf.columns:
                            data["financials"]["operating_cf"] = cf[ocf_col].head(5).tolist()
                        if capex_col in cf.columns:
                            data["financials"]["capex"] = cf[capex_col].head(5).tolist()
                        data["financials"]["cf_periods"] = cf["报告期"].head(5).tolist()
                except Exception as e:
                    logger.warning(f"Could not fetch cash flow: {e}")

                # Current valuation
                try:
                    df = ak.stock_zh_a_spot_em()
                    row = df[df["代码"] == clean_ticker]
                    if not row.empty:
                        data["valuation"] = {
                            "price": row["最新价"].values[0],
                            "change_pct": row["涨跌幅"].values[0],
                            "volume": row["成交量"].values[0],
                            "amount": row["成交额"].values[0],
                            "pe": row["市盈率-动态"].values[0] if "市盈率-动态" in row.columns else None,
                            "pb": row["市净率"].values[0] if "市净率" in row.columns else None,
                            "market_cap": row["总市值"].values[0] if "总市值" in row.columns else None,
                            "turnover": row["换手率"].values[0] if "换手率" in row.columns else None
                        }
                except Exception as e:
                    logger.warning(f"Could not fetch valuation: {e}")

                # Trading data
                try:
                    df = ak.stock_zh_a_hist(symbol=clean_ticker, period="daily", adjust="qfq")
                    if not df.empty:
                        recent = df.tail(30)
                        data["trading"] = {
                            "avg_volume_30d": recent["成交量"].mean(),
                            "avg_amount_30d": recent["成交额"].mean(),
                            "volatility_30d": recent["收盘"].pct_change().std() * (252 ** 0.5) * 100
                        }
                except Exception as e:
                    logger.warning(f"Could not fetch trading data: {e}")

            elif market == "HK":
                try:
                    df = ak.stock_hk_spot_em()
                    row = df[df["代码"] == clean_ticker]
                    if not row.empty:
                        data["info"]["name"] = row["名称"].values[0]
                        data["valuation"]["price"] = row["最新价"].values[0]
                except Exception as e:
                    logger.warning(f"Could not fetch HK stock: {e}")
                    data["info"]["name"] = clean_ticker

            elif market == "US":
                us_names = {
                    "AAPL": "Apple Inc.", "GOOGL": "Alphabet Inc.",
                    "MSFT": "Microsoft Corp.", "AMZN": "Amazon.com Inc.",
                    "TSLA": "Tesla Inc.", "NVDA": "NVIDIA Corp."
                }
                data["info"]["name"] = us_names.get(clean_ticker, clean_ticker)

        except Exception as e:
            logger.error(f"Data fetch error: {e}")
            data["info"]["name"] = clean_ticker

        ctx["stock_data"] = data
        logger.info(f"Fetched data for {data['info'].get('name', clean_ticker)}")
        return data

    def _generate_skeleton(self, ctx: dict) -> str:
        """Step 2: Generate report skeleton with placeholders"""
        data = ctx["stock_data"]
        today = datetime.now().strftime("%Y-%m-%d")
        market_names = {"A": "A股", "HK": "港股", "US": "美股"}

        info = data.get("info", {})
        financials = data.get("financials", {})
        valuation = data.get("valuation", {})
        trading = data.get("trading", {})

        def fmt(val, decimals=2, suffix=""):
            if val is None or (isinstance(val, float) and str(val) == 'nan'):
                return "-"
            try:
                v = float(val)
                if abs(v) >= 1e8:
                    return f"{v/1e8:.{decimals}f}亿{suffix}"
                elif abs(v) >= 1e4:
                    return f"{v/1e4:.{decimals}f}万{suffix}"
                else:
                    return f"{v:.{decimals}f}{suffix}"
            except:
                return str(val)

        # FCFF calculation
        fcff_rows = []
        if financials.get("operating_cf") and financials.get("capex"):
            ocf = financials["operating_cf"]
            capex = financials["capex"]
            periods = financials.get("cf_periods", [])
            for i in range(min(len(ocf), len(capex), 5)):
                try:
                    ocf_val = float(str(ocf[i]).replace("亿", "").replace("万", "")) if ocf[i] else 0
                    capex_val = float(str(capex[i]).replace("亿", "").replace("万", "")) if capex[i] else 0
                    fcff = ocf_val - capex_val
                    status = "正" if fcff > 0 else "负"
                    period = periods[i] if i < len(periods) else f"Y-{i}"
                    fcff_rows.append(f"| {period} | {fmt(ocf[i])} | {fmt(capex[i])} | {fmt(fcff)} | {status} |")
                except:
                    pass

        fcff_table = "\n".join(fcff_rows) if fcff_rows else "| {{CLAUDE_FILL}} | | | | |"

        skeleton = f'''---
tags: [stock-research, {market_names.get(data["market"], data["market"])}]
ticker: {data["ticker"]}
market: {market_names.get(data["market"], data["market"])}
company_name: {info.get("name", data["ticker"])}
created: {today}
updated: {today}
---

# {info.get("name", data["ticker"])} ({data["ticker"]}) 深度调研

## 摘要速览

- **核心逻辑**: {{{{CLAUDE_FILL:core_thesis}}}}
- **关键指标**: 市值 {fmt(valuation.get("market_cap"))} | PE {valuation.get("pe", "-")} | PB {valuation.get("pb", "-")} | ROE {{{{CLAUDE_FILL:roe}}}}%
- **自由现金流**: {{{{CLAUDE_FILL:fcff_status}}}} → 价值陷阱风险: {{{{CLAUDE_FILL:value_trap_risk}}}}
- **建议操作**: {{{{CLAUDE_FILL:action}}}} @ 目标仓位 {{{{CLAUDE_FILL:position_size}}}}%
- **适合策略**: {{{{CLAUDE_FILL:strategy}}}}
- **风险警示**: {{{{CLAUDE_FILL:key_risk}}}}

---

## 公司概况

**公司名称**: {info.get("name", data["ticker"])}
**股票代码**: {data["ticker"]}
**所属市场**: {market_names.get(data["market"], data["market"])}
**所属行业**: {info.get("industry", "{{CLAUDE_FILL}}")}

{{{{CLAUDE_FILL:company_overview}}}}

---

## 财务分析

### 关键指标摘要

| 指标 | 最新值 | 同比 | 行业均值 | 评估 |
|------|--------|------|----------|------|
| 营收 | {fmt(financials.get("revenue", [None])[0] if financials.get("revenue") else None)} | - | - | - |
| 净利润 | {fmt(financials.get("net_profit", [None])[0] if financials.get("net_profit") else None)} | - | - | - |
| 毛利率 | {financials.get("gross_margin", ["-"])[0] if financials.get("gross_margin") else "-"} | - | - | - |
| ROE | {financials.get("roe", ["-"])[0] if financials.get("roe") else "-"} | - | - | - |
| 资产负债率 | {financials.get("debt_ratio", ["-"])[0] if financials.get("debt_ratio") else "-"} | - | - | - |

### 财务质量评估

{{{{CLAUDE_FILL:financial_quality}}}}

### 自由现金流分析 (价值陷阱检测)

**计算公式**: FCFF = 经营活动现金流净额 - 资本支出

| 年份 | 经营现金流 | 资本支出 | FCFF | 判断 |
|------|-----------|---------|------|------|
{fcff_table}

**FCFF诊断结论**: {{{{CLAUDE_FILL:fcff_conclusion}}}}

---

## 业务结构

### 收入构成

{{{{CLAUDE_FILL:revenue_breakdown}}}}

### 增长驱动

{{{{CLAUDE_FILL:growth_drivers}}}}

### 护城河

- **护城河类型**: {{{{CLAUDE_FILL:moat_type}}}}
- **护城河强度**: {{{{CLAUDE_FILL:moat_width}}}}

---

## 竞争格局

{{{{CLAUDE_FILL:competitors}}}}

### 行业趋势

{{{{CLAUDE_FILL:industry_analysis}}}}

---

## 风险因素

### 风险矩阵

{{{{CLAUDE_FILL:risk_matrix}}}}

---

## 估值分析

### 估值评估

{{{{CLAUDE_FILL:valuation_assessment}}}}

- **当前价格**: {valuation.get("price", "-")}
- **合理价格区间**: {{{{CLAUDE_FILL:fair_value_range}}}}
- **安全边际**: {{{{CLAUDE_FILL:margin_of_safety}}}}

---

## 技术分析

{{{{CLAUDE_FILL:technical_conclusion}}}}

---

## 基金经理视角

| 项目 | 建议 |
|------|------|
| **操作建议** | {{{{CLAUDE_FILL:action}}}} |
| **目标仓位** | {{{{CLAUDE_FILL:position_size}}}}% |
| **入场区间** | {{{{CLAUDE_FILL:entry_price}}}} |
| **止损位** | {{{{CLAUDE_FILL:stop_loss}}}} |
| **目标价** | {{{{CLAUDE_FILL:target_price}}}} |

### 流动性分析

| 指标 | 数值 |
|------|------|
| 日均成交额 | {fmt(trading.get("avg_amount_30d"))} |
| 年化波动率 | {fmt(trading.get("volatility_30d"), 1)}% |

---

## 投资观点

### 看多理由

1. {{{{CLAUDE_FILL:bull_case_1}}}}
2. {{{{CLAUDE_FILL:bull_case_2}}}}
3. {{{{CLAUDE_FILL:bull_case_3}}}}

### 看空理由

1. {{{{CLAUDE_FILL:bear_case_1}}}}
2. {{{{CLAUDE_FILL:bear_case_2}}}}
3. {{{{CLAUDE_FILL:bear_case_3}}}}

### 综合结论

- **投资评级**: {{{{CLAUDE_FILL:rating}}}}

---

## 催化剂跟踪

{{{{CLAUDE_FILL:catalysts}}}}

---

## 信息来源

1. [东方财富](https://quote.eastmoney.com/) - {today}
2. [同花顺](https://www.10jqka.com.cn/) - {today}
'''

        ctx["skeleton"] = skeleton
        return skeleton

    async def _fill_analysis(self, ctx: dict) -> dict:
        """Step 3: Fill analysis using DeepSeek"""
        from openai import OpenAI

        stock_data = ctx["stock_data"]
        skeleton = ctx["skeleton"]

        # Create prompt
        prompt = self.ANALYSIS_PROMPT.format(
            stock_data=json.dumps(stock_data, indent=2, ensure_ascii=False, default=str)
        )

        # Call DeepSeek API
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )

        logger.info(f"Calling {self.model} for analysis...")

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是专业的基金经理，擅长股票分析。请严格按照JSON格式返回分析结果。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )

        content = response.choices[0].message.content

        # Parse JSON from response
        try:
            # Find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Response: {content[:500]}")
            raise

        # Fill placeholders
        filled_report = skeleton

        # Simple replacements
        replacements = {
            "{{CLAUDE_FILL:core_thesis}}": analysis.get("core_thesis", "待分析"),
            "{{CLAUDE_FILL:roe}}": analysis.get("roe", "-"),
            "{{CLAUDE_FILL:fcff_status}}": analysis.get("fcff_status", "待分析"),
            "{{CLAUDE_FILL:value_trap_risk}}": analysis.get("value_trap_risk", "中"),
            "{{CLAUDE_FILL:action}}": analysis.get("action", "持有"),
            "{{CLAUDE_FILL:position_size}}": analysis.get("position_size", "3"),
            "{{CLAUDE_FILL:strategy}}": analysis.get("strategy", "波段操作"),
            "{{CLAUDE_FILL:key_risk}}": analysis.get("key_risk", "待分析"),
            "{{CLAUDE_FILL:company_overview}}": analysis.get("company_overview", "待分析"),
            "{{CLAUDE_FILL:financial_quality}}": analysis.get("financial_quality", "待分析"),
            "{{CLAUDE_FILL:fcff_conclusion}}": analysis.get("fcff_conclusion", "待分析"),
            "{{CLAUDE_FILL:revenue_breakdown}}": analysis.get("revenue_breakdown", "待分析"),
            "{{CLAUDE_FILL:growth_drivers}}": analysis.get("growth_drivers", "待分析"),
            "{{CLAUDE_FILL:moat_type}}": analysis.get("moat_type", "待分析"),
            "{{CLAUDE_FILL:moat_width}}": analysis.get("moat_width", "待分析"),
            "{{CLAUDE_FILL:competitors}}": analysis.get("competitors", "待分析"),
            "{{CLAUDE_FILL:industry_analysis}}": analysis.get("industry_analysis", "待分析"),
            "{{CLAUDE_FILL:risk_matrix}}": analysis.get("risk_matrix", "待分析"),
            "{{CLAUDE_FILL:valuation_assessment}}": analysis.get("valuation_assessment", "待分析"),
            "{{CLAUDE_FILL:fair_value_range}}": analysis.get("fair_value_range", "待分析"),
            "{{CLAUDE_FILL:margin_of_safety}}": analysis.get("margin_of_safety", "-"),
            "{{CLAUDE_FILL:technical_conclusion}}": analysis.get("technical_conclusion", "待分析"),
            "{{CLAUDE_FILL:entry_price}}": analysis.get("entry_price", "-"),
            "{{CLAUDE_FILL:stop_loss}}": analysis.get("stop_loss", "-"),
            "{{CLAUDE_FILL:target_price}}": analysis.get("target_price", "-"),
            "{{CLAUDE_FILL:rating}}": analysis.get("rating", "中性"),
            "{{CLAUDE_FILL:catalysts}}": analysis.get("catalysts", "待分析"),
        }

        # Handle bull/bear cases
        bull_case = analysis.get("bull_case", ["待分析", "待分析", "待分析"])
        bear_case = analysis.get("bear_case", ["待分析", "待分析", "待分析"])

        if isinstance(bull_case, list) and len(bull_case) >= 3:
            replacements["{{CLAUDE_FILL:bull_case_1}}"] = bull_case[0]
            replacements["{{CLAUDE_FILL:bull_case_2}}"] = bull_case[1]
            replacements["{{CLAUDE_FILL:bull_case_3}}"] = bull_case[2]

        if isinstance(bear_case, list) and len(bear_case) >= 3:
            replacements["{{CLAUDE_FILL:bear_case_1}}"] = bear_case[0]
            replacements["{{CLAUDE_FILL:bear_case_2}}"] = bear_case[1]
            replacements["{{CLAUDE_FILL:bear_case_3}}"] = bear_case[2]

        for placeholder, value in replacements.items():
            filled_report = filled_report.replace(placeholder, str(value))

        # Clean up any remaining generic placeholders
        filled_report = re.sub(r'\{\{CLAUDE_FILL[^}]*\}\}', '待分析', filled_report)

        ctx["filled_report"] = filled_report
        ctx["analysis"] = analysis
        return analysis

    def _write_report(self, ctx: dict) -> Path:
        """Step 4: Write report to file"""
        market = ctx["market"]
        ticker = ctx["clean_ticker"]
        report = ctx["filled_report"]

        # Create output directory
        output_dir = self.output_dir / market
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write file
        output_path = output_dir / f"{ticker}.md"
        output_path.write_text(report, encoding="utf-8")

        ctx["output_path"] = output_path
        logger.info(f"Report written to: {output_path}")
        return output_path

    def _quality_check(self, ctx: dict) -> dict:
        """Step 5: Verify report quality"""
        output_path = ctx["output_path"]
        checks = {
            "file_exists": False,
            "no_placeholders": False,
            "has_rating": False,
            "has_prices": False,
            "passed": False
        }

        if not output_path.exists():
            return checks

        content = output_path.read_text(encoding="utf-8")
        checks["file_exists"] = True

        # Check for remaining placeholders
        placeholder_count = len(re.findall(r'\{\{CLAUDE_FILL', content))
        checks["no_placeholders"] = placeholder_count == 0
        if placeholder_count > 0:
            logger.warning(f"Report has {placeholder_count} unfilled placeholders")

        # Check for rating
        checks["has_rating"] = "投资评级" in content and "待分析" not in content.split("投资评级")[1][:50]

        # Check for prices
        checks["has_prices"] = "入场区间" in content and "止损位" in content

        # Overall pass
        checks["passed"] = checks["file_exists"] and checks["no_placeholders"]

        return checks


async def research_stock(
    ticker: str,
    output_dir: str = "./stock-research",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model: str = "deepseek-chat"
) -> WorkflowResult:
    """
    Convenience function to run stock research workflow.

    Args:
        ticker: Stock ticker (e.g., 601288, 00700, AAPL)
        output_dir: Directory to save reports
        api_key: DeepSeek API key (or set DEEPSEEK_API_KEY env)
        api_base: API base URL (default: https://api.deepseek.com)
        model: Model to use (default: deepseek-chat)

    Returns:
        WorkflowResult with success status and output path
    """
    workflow = StockResearchWorkflow(
        output_dir=Path(output_dir),
        api_key=api_key,
        api_base=api_base,
        model=model
    )

    return await workflow.run(ticker=ticker)
