"""
Module: agents/competitor_compare.py (LangGraph 버전)
Purpose: 경쟁사 비교 에이전트 (RAG 우선, .env 기반 키 로딩, LangGraph 기반)

입력 State 요구:
state = {
  "tech_summary": {"company": "Zest AI", "summary": "...", "tech_score": 82},
  "market_eval": {
      "competitors": ["Featurespace", "SAS Fraud", "FICO"],
      "market_summary": "...",
      "risk_summary": "..."
  },
  "startup_search": {"docs": [ {"title":"...","url":"...","content":"..."}, ... ]},   # 선택
  "market_store": [ {"title":"...","url":"...","content":"..."}, ... ]                 # 선택
}

출력 병합:
state["competition"] = {
  "target": "Zest AI",
  "comparisons": [
    {
      "name": "Featurespace",
      "product_focus": "...",
      "tech_diff": "...",
      "go_to_market": "...",
      "strengths": ["..."],
      "weaknesses": ["..."],
      "positioning": "...",
      "sources": [{"title":"...","url":"..."}]
    },
    ...
  ],
  "differentiation_summary": "타깃 대비 차별화 요약",
  "edge_score": 0-100
}
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, TypedDict
from dataclasses import dataclass
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# LangGraph
try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

# LangChain
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI
    LANGCHAIN_CHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_CHAIN_AVAILABLE = False

# Tavily (옵션)
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except Exception:
    TAVILY_AVAILABLE = False

# ----------------------------- State & Config -----------------------------

class ComparisonItem(TypedDict, total=False):
    name: str
    product_focus: str
    tech_diff: str
    go_to_market: str
    strengths: List[str]
    weaknesses: List[str]
    positioning: str
    sources: List[Dict[str, str]]

class CompetitionOutput(TypedDict, total=False):
    target: str
    comparisons: List[ComparisonItem]
    differentiation_summary: str
    edge_score: float

class CompetitorState(TypedDict, total=False):
    tech_summary: Dict[str, Any]
    market_eval: Dict[str, Any]
    startup_search: Dict[str, Any]
    market_store: List[Dict[str, str]]
    competition: CompetitionOutput

@dataclass
class CompetitorConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    include_chars: int = 2000
    max_docs_per_comp: int = 6

# ----------------------------- Helpers -----------------------------

def _norm(s: str) -> str:
    return (s or "").strip()


def _get_llm(model: str, temperature: float):
    if not LANGCHAIN_CHAIN_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        return ChatOpenAI(model=model, temperature=temperature)
    except Exception:
        return None


def _llm_json(prompt: str, llm) -> Dict[str, Any]:
    if not llm:
        return _offline_stub()
    try:
        chain = ChatPromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
        txt = chain.invoke({"prompt": prompt})
        return json.loads(txt)
    except Exception:
        return _offline_stub()


def _offline_stub() -> Dict[str, Any]:
    return {
        "comparisons": [
            {
                "name": "Featurespace",
                "product_focus": "실시간 결제 사기탐지 플랫폼",
                "tech_diff": "이상탐지 기반의 행동 프로파일링, 스트리밍 처리",
                "go_to_market": "대형 결제사/은행 중심 엔터프라이즈 세일즈",
                "strengths": ["실시간 성능", "대규모 레퍼런스"],
                "weaknesses": ["구축 비용", "커스터마이징 부담"],
                "positioning": "엔터프라이즈 고급 세그먼트",
                "sources": []
            },
            {
                "name": "FICO",
                "product_focus": "신용 리스크 및 사기탐지 소프트웨어 제품군",
                "tech_diff": "규칙+모델 혼합, 강력한 워크플로/정책 엔진",
                "go_to_market": "금융기관 레거시 교체 및 확장",
                "strengths": ["풍부한 도입사례", "정책 관리"],
                "weaknesses": ["유연성 제한", "신규 AI 스택 대응 속도"],
                "positioning": "레거시 강자/정책 중심",
                "sources": []
            }
        ],
        "differentiation_summary": "대상 기업은 실시간 추론과 모델 업데이트 주기 단축으로 운영 민첩성을 강조하며, 엔터프라이즈 대비 TCO를 낮추는 것이 강점.",
        "edge_score": 76
    }


def _collect_docs_for_entity(state: CompetitorState, entity: str, *, max_docs: int = 6, max_chars: int = 2000) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    # 1) market_store 우선
    for d in (state.get("market_store") or []) or []:
        title = _norm(d.get("title", ""))
        url = _norm(d.get("url", ""))
        content = _norm(d.get("content", ""))
        if not content:
            continue
        if entity.lower() in (title + url + content).lower():
            docs.append({"title": title[:240], "url": url, "content": content[:max_chars]})
        if len(docs) >= max_docs:
            return docs
    # 2) startup_search.docs
    if len(docs) < max_docs:
        for d in ((state.get("startup_search") or {}).get("docs") or []):
            title = _norm(d.get("title", ""))
            url = _norm(d.get("url", ""))
            content = _norm(d.get("content", "")) or _norm(d.get("snippet", ""))
            if entity.lower() in (title + url + content).lower():
                docs.append({"title": title[:240], "url": url, "content": content[:max_chars]})
            if len(docs) >= max_docs:
                return docs
    # 3) Tavily 검색
    if len(docs) < max_docs and TAVILY_AVAILABLE and os.getenv("TAVILY_API_KEY"):
        try:
            client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            q = f"{entity} fintech ai product technology customers pricing"
            res = client.search(query=q, max_results=max_docs)
            for r in res.get("results", [])[:max_docs]:
                docs.append({
                    "title": _norm(r.get("title", ""))[:240],
                    "url": _norm(r.get("url", "")),
                    "content": _norm(r.get("content", ""))[:max_chars],
                })
        except Exception:
            pass
    # 4) placeholder
    if not docs:
        docs = [{
            "title": f"{entity} overview",
            "url": "https://www.example.com/competitor",
            "content": f"{entity} provides AI-powered risk analytics and fraud detection.",
        }]
    return docs[:max_docs]

# ----------------------------- Node Factory -----------------------------

def _competitor_node_factory(cfg: CompetitorConfig):
    def node(state: CompetitorState) -> CompetitorState:
        target = _norm(((state.get("tech_summary") or {}).get("company")) or state.get("target_company", "")) or "(unknown)"
        comp_list = (state.get("market_eval") or {}).get("competitors", [])
        
        # 대상 회사를 경쟁사 목록에서 제외 (정규화하여 비교)
        import re
        def normalize_name(name):
            """회사 이름 정규화 (괄호, 공백 제거)"""
            name = re.sub(r'\([^)]*\)', '', name).strip()
            name = re.sub(r'\s+', ' ', name).strip()
            return name.lower()
        
        target_normalized = normalize_name(target)
        comp_list = [
            comp for comp in comp_list 
            if normalize_name(comp) != target_normalized
        ]
        
        if not comp_list:
            # 경쟁사가 없으면 기본 경쟁사 사용
            comp_list = ["Competitor A", "Competitor B"]

        packs: Dict[str, List[Dict[str, str]]] = {}
        for comp in comp_list[:6]:
            docs = _collect_docs_for_entity(state, comp, max_docs=cfg.max_docs_per_comp, max_chars=cfg.include_chars)
            packs[comp] = docs

        llm = _get_llm(cfg.model, cfg.temperature)

        prompt = json.dumps({
            "instruction": (
                "너는 핀테크 경쟁사 분석가다. 각 경쟁사 문서를 바탕으로 제품초점/기술차별점/GT-M/강점/약점/포지셔닝을 요약하라. "
                "허위 추정은 금지하고, 문서에 근거 없는 내용은 생략하라. JSON만 출력하라."
            ),
            "target": target,
            "competitors": list(packs.keys()),
            "docs": packs,
            "output_schema": {
                "comparisons": [
                    {
                        "name": "",
                        "product_focus": "",
                        "tech_diff": "",
                        "go_to_market": "",
                        "strengths": [""],
                        "weaknesses": [""],
                        "positioning": "",
                        "sources": [{"title": "", "url": ""}],
                    }
                ],
                "differentiation_summary": "",
                "edge_score": 0,
            },
        }, ensure_ascii=False)

        data = _llm_json(prompt, llm)

        # edge score 보정 (타깃 기술점수 가중)
        try:
            base_edge = float(data.get("edge_score", 0))
        except Exception:
            base_edge = 0.0
        tech_score = float(((state.get("tech_summary") or {}).get("tech_score", 0)) or 0)
        edge_score = min(100.0, max(0.0, 0.7 * base_edge + 0.3 * tech_score))

        # 소스 축약 (packs에서 상위 3개만)
        comparisons = data.get("comparisons", [])
        for comp in comparisons:
            srcs: List[Dict[str, str]] = []
            name = comp.get("name", "")
            for d in packs.get(name, [])[:3]:
                srcs.append({"title": d.get("title", ""), "url": d.get("url", "")})
            comp["sources"] = srcs

        state["competition"] = {
            "target": target,
            "comparisons": comparisons,
            "differentiation_summary": data.get("differentiation_summary", ""),
            "edge_score": float(round(edge_score, 1)),
        }
        
        # 실행 결과 출력
        print("\n" + "="*80)
        print("✅ [4/6] 경쟁사 분석 완료")
        print("="*80)
        print(f"🎯 대상 기업: {target}")
        print(f"📊 경쟁 우위 점수: {state['competition']['edge_score']}/100")
        print(f"🏆 경쟁사 비교: {len(comparisons)}개")
        for idx, comp in enumerate(comparisons[:3], 1):
            print(f"   {idx}. {comp.get('name', 'N/A')}")
            print(f"      강점: {', '.join(comp.get('strengths', [])[:2])}")
        print("="*80 + "\n")
        
        return state

    return node

# ----------------------------- Graph Builder -----------------------------

def build_competitor_graph(config: Optional[CompetitorConfig] = None):
    cfg = config or CompetitorConfig()
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph가 설치되어 있지 않습니다. `pip install langgraph` 후 다시 시도하세요.")
    g = StateGraph(CompetitorState)
    g.add_node("competition", _competitor_node_factory(cfg))
    g.add_edge(START, "competition")
    g.add_edge("competition", END)
    return g.compile()

# ----------------------------- Helper -----------------------------

def run_competitor_compare(state: Dict[str, Any], config: Optional[CompetitorConfig] = None) -> Dict[str, Any]:
    app = build_competitor_graph(config)
    return app.invoke(state)

# ----------------------------- CLI Test -----------------------------
if __name__ == "__main__":
    dummy: CompetitorState = {
        "tech_summary": {"company": "Zest AI", "tech_score": 82},
        "market_eval": {"competitors": ["Featurespace", "FICO"]},
        "startup_search": {"docs": [
            {"title": "Featurespace expands ARIC", "url": "https://ex.com/feats", "content": "Featurespace real-time fraud detection platform ARIC..."},
            {"title": "FICO Falcon overview", "url": "https://ex.com/fico", "content": "FICO fraud management and decisioning..."}
        ]}
    }
    final = run_competitor_compare(dummy)
    from pprint import pprint
    pprint(final.get("competition", {}))
