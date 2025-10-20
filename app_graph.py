"""
App: app_graph.py
Purpose: LangGraph 통합 파이프라인 (완전 통합 버전)
순서: startup_search → tech_summary → market_evaluation → competitor_compare → investment_decision → report_generator

특징:
- 각 에이전트가 필요한 데이터를 이전 에이전트에서 자동으로 받아옴
- tech_summary에서 CompanyDetail 생성 → market_evaluation에서 사용
- 가중치 기반 투자 판단 (fintech 방식)
- PDF + Markdown 보고서 생성

사용 예시 (CLI):
    python app_graph.py --segment "fintech ai" --region Korea --limit 5 --print-report

사용 예시 (Notebook):
    from app_graph import build_pipeline, run_pipeline
    app = build_pipeline()
    final = run_pipeline(app, {"segment":"fintech ai", "region":"Korea", "limit":5})
    final["investment"], final["report"]["pdf_path"]
"""
from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, Optional, TypedDict
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# OpenMP 중복 로드 오류 방지
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# LangGraph
from langgraph.graph import StateGraph, START, END

# Agents (LangGraph 버전 모듈)
from agents.startup_search import build_startup_search_graph, StartupSearchConfig
from agents.tech_summary import build_tech_summary_graph, TechSummaryConfig
from agents.market_evaluation import build_market_eval_graph, MarketEvalConfig
from agents.competitor_compare import build_competitor_graph, CompetitorConfig
from agents.investment_decision import build_investment_graph, InvestmentConfig
from agents.report_generator import build_report_graph, ReportConfig, save_report_complete


# ----------------------------- Master State -----------------------------
class MasterState(TypedDict, total=False):
    # 입력 파라미터
    segment: str
    region: str
    limit: int
    language: str
    target_company: str
    
    # 데이터 경로
    report_dir: str  # PDF 리포트 폴더 (data)
    market_store: list
    
    # 각 에이전트 출력
    startup_search: dict
    tech_summary: dict
    market_eval: dict
    competition: dict
    investment: dict
    report: dict


# ----------------------------- Utilities -----------------------------

def _maybe_load_market_store(path: Optional[str]) -> Optional[list]:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except Exception:
        return None


def _subgraph_node(app):
    """컴파일된 LangGraph 서브그래프를 래핑하여 노드처럼 사용."""
    def node(state: MasterState) -> MasterState:
        return app.invoke(state)
    return node


# ----------------------------- Pipeline Builder -----------------------------

def build_pipeline(
    *,
    startup_cfg: Optional[StartupSearchConfig] = None,
    tech_cfg: Optional[TechSummaryConfig] = None,
    market_cfg: Optional[MarketEvalConfig] = None,
    comp_cfg: Optional[CompetitorConfig] = None,
    inv_cfg: Optional[InvestmentConfig] = None,
    report_cfg: Optional[ReportConfig] = None,
    use_fintech_mode: bool = True,  # fintech 통합 모드 사용
):
    """
    LangGraph 파이프라인 구축
    
    순서: startup_search → tech_summary → market_evaluation → 
          competitor_compare → investment_decision → report_generator
    """
    # 기본 설정값 (fintech 모드)
    if startup_cfg is None:
        startup_cfg = StartupSearchConfig(
            use_structured_output=True,  # LLM structured output
            model="gpt-4o-mini",
            temperature=0.1
        )
    
    if tech_cfg is None:
        tech_cfg = TechSummaryConfig(
            use_structured_output=True,  # CompanyDetail 생성
            collect_web_info=True,  # 웹 검색으로 기술 정보 수집
            temperature=0.0
        )
    
    if market_cfg is None:
        market_cfg = MarketEvalConfig(
            use_agentic_rag=True,  # Agentic RAG 사용 (Query Rewrite + Relevance Check + Web Fallback)
            # report_dir는 __post_init__에서 자동 설정 (data)
        )
    
    if comp_cfg is None:
        comp_cfg = CompetitorConfig()
    
    if inv_cfg is None:
        inv_cfg = InvestmentConfig(
            use_fintech_weights=use_fintech_mode,  # fintech 가중치 방식
            invest_threshold=3.5
        )
    
    if report_cfg is None:
        report_cfg = ReportConfig(
            generate_pdf=True,  # PDF 생성
            output_dir="outputs"
        )
    
    # 하위 그래프(서브그래프) 컴파일
    sg_search = build_startup_search_graph(startup_cfg)
    sg_tech = build_tech_summary_graph(tech_cfg)
    sg_market = build_market_eval_graph(market_cfg)
    sg_comp = build_competitor_graph(comp_cfg)
    sg_invest = build_investment_graph(inv_cfg)
    sg_report = build_report_graph(report_cfg)

    # 마스터 그래프 구성
    g = StateGraph(MasterState)
    g.add_node("startup_search", _subgraph_node(sg_search))
    g.add_node("tech_summary", _subgraph_node(sg_tech))
    g.add_node("market_eval", _subgraph_node(sg_market))
    g.add_node("competition", _subgraph_node(sg_comp))
    g.add_node("investment", _subgraph_node(sg_invest))
    g.add_node("report", _subgraph_node(sg_report))

    # 순차적 연결: startup_search → tech_summary → market_evaluation → 
    #             competitor_compare → investment_decision → report_generator
    g.add_edge(START, "startup_search")
    g.add_edge("startup_search", "tech_summary")
    g.add_edge("tech_summary", "market_eval")
    g.add_edge("market_eval", "competition")
    g.add_edge("competition", "investment")
    g.add_edge("investment", "report")
    g.add_edge("report", END)

    return g.compile()


# ----------------------------- Runner -----------------------------

def run_pipeline(app, state: Dict[str, Any]) -> Dict[str, Any]:
    return app.invoke(state)


# ----------------------------- CLI -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LangGraph FinTech AI Investment Pipeline (완전 통합)")
    p.add_argument("--segment", type=str, default="fintech ai", help="탐색/시장 키워드")
    p.add_argument("--region", type=str, default="Korea", help="지역 (Korea|Global)")
    p.add_argument("--limit", type=int, default=5, help="탐색 후보 최대 개수")
    p.add_argument("--target-company", type=str, default=None, help="특정 회사 고정")
    p.add_argument("--report-dir", type=str, default=None, help="PDF 리포트 폴더 (기본: data)")
    p.add_argument("--use-llm-exec-summary", action="store_true", help="보고서 Executive Summary 간결화(LLM)")
    p.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM 모델명")
    p.add_argument("--out-dir", type=str, default="outputs", help="보고서 저장 폴더")
    p.add_argument("--print-report", action="store_true", help="생성된 마크다운 출력")
    p.add_argument("--legacy-mode", action="store_true", help="레거시 모드 사용 (0-100점 척도)")
    return p.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("🚀 AI 핀테크 투자 분석 파이프라인 (LangGraph)")
    print("=" * 80)
    print("\n📋 실행 설정:")
    print(f"  - 세그먼트: {args.segment}")
    print(f"  - 지역: {args.region}")
    print(f"  - 스타트업 수: {args.limit}")
    print(f"  - 평가 방식: {'레거시 (0-100점)' if args.legacy_mode else 'fintech (가중치 기반)'}")
    
    # 초기 상태 구성
    state: MasterState = {
        "segment": args.segment,
        "region": args.region,
        "limit": args.limit,
        "language": "ko",
    }
    
    # report_dir 설정 (data 기본값)
    if args.report_dir:
        state["report_dir"] = args.report_dir
    else:
        # 기본값: app_graph.py 기준 data 폴더
        current_file = Path(__file__).resolve()
        state["report_dir"] = str(current_file.parent / "data")
    
    if args.target_company:
        state["target_company"] = args.target_company
    
    print(f"  - PDF 리포트 경로: {state['report_dir']}")
    print(f"  - 보고서 출력: {args.out_dir}")

    # 파이프라인 빌드 (기본 설정 사용)
    app = build_pipeline(use_fintech_mode=not args.legacy_mode)
    
    print("\n" + "=" * 80)
    print("⏳ 파이프라인 실행 중...")
    print("=" * 80)

    # 실행
    final = run_pipeline(app, state)

    # 결과 저장
    print("\n" + "=" * 80)
    print("💾 보고서 저장 중...")
    print("=" * 80)
    
    try:
        saved_paths = save_report_complete(final, output_dir=args.out_dir)
        print(f"\n✅ 저장 완료:")
        for k, v in saved_paths.items():
            print(f"  - {k}: {v}")
    except Exception as e:
        print(f"⚠️ 저장 중 오류: {e}")

    # 결과 요약 출력
    print("\n" + "=" * 80)
    print("📊 최종 결과 요약")
    print("=" * 80)
    
    # 각 단계별 결과
    search = final.get("startup_search", {})
    tech = final.get("tech_summary", {})
    market = final.get("market_eval", {})
    comp = final.get("competition", {})
    inv = final.get("investment", {})
    rep = final.get("report", {})
    
    print(f"\n1️⃣ 스타트업 검색: {len(search.get('candidates', []))}개 발견")
    if search.get('candidates'):
        print(f"   - 대표: {search['candidates'][0].get('name', 'N/A')}")
    
    print(f"\n2️⃣ 기술 요약:")
    print(f"   - 회사: {tech.get('company', 'N/A')}")
    print(f"   - 기술 점수: {tech.get('tech_score', 0):.1f}")
    print(f"   - 제품 수: {len(tech.get('products', []))}")
    
    print(f"\n3️⃣ 시장 평가:")
    print(f"   - 시장 점수: {market.get('market_score', 0):.1f}")
    print(f"   - 리스크 점수: {market.get('risk_score', 0):.1f}")
    
    print(f"\n4️⃣ 경쟁 분석:")
    print(f"   - 경쟁 우위 점수: {comp.get('edge_score', 0):.1f}")
    
    print(f"\n5️⃣ 투자 판단:")
    print(f"   - 최종 점수: {inv.get('weighted_score', inv.get('total_score', 0)):.2f}")
    print(f"   - 결정: {inv.get('decision', 'N/A')}")
    
    print(f"\n6️⃣ 보고서:")
    print(f"   - 파일명: {rep.get('filename', 'N/A')}")
    if rep.get('pdf_path'):
        print(f"   - PDF: {rep['pdf_path']}")
    
    print("\n" + "=" * 80)

    if args.print_report:
        print("\n📄 마크다운 보고서:")
        print("=" * 80)
        print(rep.get("markdown", ""))
        print("=" * 80)


if __name__ == "__main__":
    main()
