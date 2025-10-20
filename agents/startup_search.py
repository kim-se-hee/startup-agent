"""
Module: agents/startup_search.py (LangGraph + 한국 AI 핀테크 특화 버전)
Purpose: "스타트업 탐색" 에이전트 (LLM Structured Output 기반) — LangGraph StateGraph

특징:
- 한국 AI 핀테크 스타트업 특화 검색
- LLM Structured Output으로 정확한 데이터 추출
- Pydantic 스키마 기반 검증
- 중복 제거 및 관련도 스코어링

입력 State 예시:
state = {
    "segment": "fintech ai",        # (필수) 탐색 도메인/키워드
    "region": "Korea",              # (선택) 지역 (Korea|Global)
    "limit": 10,                    # (선택) 후보 수 상한
    "language": "ko"                # (선택) 언어
}

출력 병합 예시:
state["startup_search"] = {
  "query": "...",
  "candidates": [
      {"name": "토스", "domain": "Fintech/결제", "description": "...", "url": "...", "score": 0.95},
      ...
  ],
  "docs": [...],
  "ts": 1690000000
}

환경 변수:
- OPENAI_API_KEY (필수)
- TAVILY_API_KEY (필수)
"""
from __future__ import annotations

import os
import re
import time
import hashlib
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from urllib.parse import urlparse, urlunparse
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# LangGraph
try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

# LangChain (Structured Output용)
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# Pydantic (스키마)
try:
    from pydantic import BaseModel, Field, ConfigDict
    PYDANTIC_AVAILABLE = True
except Exception:
    PYDANTIC_AVAILABLE = False

# Tavily (필수)
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except Exception as e:
    TAVILY_AVAILABLE = False
    print(f"⚠️ Tavily import 실패: {e}")

# ----------------------------- Pydantic Models (from single file) -----------------------------

if PYDANTIC_AVAILABLE:
    class StartupHit(BaseModel):
        """스타트업 검색 결과"""
        name: str
        domain: str
        description: str
        url: Optional[str] = ""
        score: Optional[float] = 0.0
        
        model_config = ConfigDict(str_strip_whitespace=True)
    
    class StartupSearchResult(BaseModel):
        """LLM 구조화 출력용"""
        items: List[StartupHit] = []
else:
    StartupHit = dict
    StartupSearchResult = dict


# ----------------------------- State & Config -----------------------------

class StartupSearchOutput(TypedDict, total=False):
    query: str
    candidates: List[Dict[str, Any]]
    docs: List[Dict[str, Any]]
    ts: int

class StartupSearchState(TypedDict, total=False):
    segment: str
    region: str
    limit: int
    language: str
    startup_search: StartupSearchOutput

@dataclass
class StartupSearchConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_results: int = 10
    include_content_chars: int = 1800
    use_structured_output: bool = True  # LLM structured output 사용 여부

# ----------------------------- Utilities -----------------------------

FINTECH_KEYWORDS = [
    "fintech", "payment", "payments", "lending", "credit", "scoring",
    "fraud", "risk", "insurance", "insurtech", "kyc", "aml", "banking",
]
AI_KEYWORDS = ["ai", "machine learning", "ml", "deep learning", "llm", "model"]

def _norm(text: str) -> str:
    return (text or "").strip()

def normalize_root_url(url: str) -> str:
    """URL 정규화 (from single file)"""
    try:
        u = urlparse(url)
        clean = u._replace(path="/", params="", query="", fragment="")
        return urlunparse(clean)
    except Exception:
        return url

def _score(title: str, snippet: str) -> float:
    text = f"{title} {snippet}".lower()
    f_hits = sum(k in text for k in FINTECH_KEYWORDS)
    a_hits = sum(k in text for k in AI_KEYWORDS)
    return min(1.0, (0.6 * (f_hits/5.0)) + (0.4 * (a_hits/4.0)))

def _extract_company_from_title(title: str) -> str:
    t = title.split(" - ")[0].split(" | ")[0]
    t = re.sub(r"(News|TechCrunch|Crunchbase|Forbes|Bloomberg|Official Site)$", "", t, flags=re.I).strip()
    t = re.sub(r"(AI|Fintech|Startup|Company)$", "", t, flags=re.I).strip()
    return t if len(t) >= 2 else title.strip()

def _dedupe_by_url(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        key = hashlib.md5(_norm(it.get("url", "")).encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# ----------------------------- Prompts (from single file) -----------------------------

SYSTEM_PROMPT = """당신은 한국 스타트업 전문 리서처입니다.
목표: 2025년 기준, '한국의 AI 기술을 사용하는 핀테크(금융) 스타트업'만 추려
'name, domain, description' 필드로 구성된 JSON을 반환합니다.

규칙:
- **한국 기업만 선정**: 한국에 본사를 둔 스타트업만 포함. 해외 기업은 제외.
- 반드시 'AI 기술을 핵심에 활용'하는 핀테크일 것(LLM/RAG/ML/NLP/CV/추천/리스크모델 등).
- 은행/대기업의 사업부나 BaaS 벤더는 제외하고 '스타트업' 중심.
- 블록체인/가상자산은 '핀테크'로 간주 가능하나, AI 핵심 활용이 확인돼야 포함.
- 2025년 맥락(최근 기사/공식 소개)을 우선 반영. 오래된/비활성 회사 제외.
- 회사당 1~2문장으로 description을 작성(한국어).
- domain은 'Fintech/세부분야' 형태로 간결하게 표기(예: 'Fintech/신용평가', 'Fintech/결제').
- 중복/동일 회사 제거, 최대 {limit}개.
- 최종 출력은 JSON만(문장·해설 금지), 스키마:
  {{"items":[{{"name":"","domain":"","description":""}}, ...]}}
"""

USER_QUERY_TMPL = """아래는 웹 검색 결과입니다. 2025년 기준으로 유효한 '한국 AI 핀테크 스타트업'만 추려 주세요.
해외 기업은 제외하고, 한국 기업만 선정하세요.
한국어로 요약하며, 지정 스키마(JSON-only)로만 답하세요.

검색 질의:
{query}

검색 결과(최대 30개):
{results}
"""

# ----------------------------- Node (LangGraph) -----------------------------

def _startup_search_node_factory(cfg: StartupSearchConfig):
    """LangGraph 노드 함수를 구성하는 팩토리. cfg를 클로저로 캡처."""
    def node(state: StartupSearchState) -> StartupSearchState:
        segment = _norm(state.get("segment", "fintech ai"))
        region = _norm(state.get("region", "Korea"))
        limit = int(state.get("limit", cfg.max_results))
        language = _norm(state.get("language", "ko"))

        # 한국 스타트업에 특화된 검색어 (from single file)
        if region.lower() == "korea":
            base_terms = [
                "한국 AI 신용평가 스타트업",
                "국내 로보어드바이저 스타트업",
                "한국 AI 대출 핀테크",
                "국내 이상거래탐지 FDS AI",
                "한국 핀테크 스타트업 투자유치",
                "국내 금융 AI 스타트업 시리즈",
                "크레파스 뱅크샐러드 토스",
                "핀테크 스타트업 AI 기술 활용",
                "금융 인공지능 스타트업 사례"
            ]
        else:
            base_terms = [
                f"{segment} {region} startup",
                "Korean fintech AI companies",
                "South Korea fintech artificial intelligence"
            ]
        
        query = " OR ".join(base_terms)
        docs: List[Dict[str, Any]] = []
        search_results_raw = []

        # Tavily 검색
        tavily_key = os.getenv("TAVILY_API_KEY")
        tavily_success = False
        
        if TAVILY_AVAILABLE and tavily_key:
            try:
                print(f"🔍 Tavily 검색 시작: {base_terms[0]}")
                client = TavilyClient(api_key=tavily_key)
                
                # 여러 쿼리로 나눠서 검색 (더 많은 결과 확보)
                for term in base_terms[:2]:  # 처음 2개 검색어만 사용
                    try:
                        res = client.search(
                            query=term,
                            max_results=min(limit * 2, 20),
                            search_depth="advanced"  # 더 깊은 검색
                        )
                        
                        results = res.get("results", [])
                        print(f"   ✅ '{term}': {len(results)}개 결과 발견")
                        
                        for r in results:
                            doc = {
                                "source": "tavily",
                                "url": _norm(r.get("url")),
                                "title": _norm(r.get("title")),
                                "snippet": _norm(r.get("content", ""))[:240],
                                "content": _norm(r.get("content", ""))[: cfg.include_content_chars],
                            }
                            docs.append(doc)
                            search_results_raw.append(doc)
                        
                        if docs:
                            tavily_success = True
                            
                    except Exception as e:
                        print(f"   ⚠️ '{term}' 검색 실패: {e}")
                        continue
                
            except Exception as e:
                print(f"⚠️ Tavily 클라이언트 초기화 실패: {e}")
        else:
            if not tavily_key:
                print(f"⚠️ TAVILY_API_KEY가 설정되지 않았습니다.")
            if not TAVILY_AVAILABLE:
                print(f"⚠️ Tavily 라이브러리가 설치되지 않았습니다. (pip install tavily-python)")

        # Tavily 실패 시 더 다양한 placeholder 데이터 제공
        if not docs:
            print(f"⚠️ 검색 결과 없음 - 기본 한국 AI 핀테크 데이터 사용")
            docs = [
                {
                    "source": "placeholder",
                    "url": "https://toss.im",
                    "title": "토스 - AI 기반 금융 슈퍼앱",
                    "snippet": "머신러닝 기반 신용평가, 맞춤형 금융상품 추천, 사기탐지 시스템",
                    "content": "토스는 AI/ML 기술을 활용하여 신용평가, 개인화 추천, 사기탐지 등 다양한 금융 서비스를 제공하는 한국의 대표 핀테크 기업입니다.",
                },
                {
                    "source": "placeholder",
                    "url": "https://www.banksalad.com",
                    "title": "뱅크샐러드 - AI 자산관리 플랫폼",
                    "snippet": "AI 기반 자산관리, 금융상품 비교 추천, 개인화된 재무 분석",
                    "content": "뱅크샐러드는 AI 기술을 활용하여 사용자의 금융 데이터를 분석하고 맞춤형 자산관리 서비스를 제공합니다.",
                },
                {
                    "source": "placeholder",
                    "url": "https://www.8percent.kr",
                    "title": "8퍼센트 - AI 기반 P2P 금융",
                    "snippet": "머신러닝 신용평가 모델, 리스크 관리 시스템",
                    "content": "8퍼센트는 AI 기반 신용평가 시스템을 통해 P2P 대출 서비스를 제공하는 핀테크 기업입니다.",
                },
                {
                    "source": "placeholder",
                    "url": "https://www.rainist.com",
                    "title": "레이니스트(뱅크샐러드) - 금융 데이터 분석",
                    "snippet": "AI 기반 금융 데이터 분석, 맞춤형 금융상품 추천",
                    "content": "레이니스트는 뱅크샐러드를 운영하며 AI 기술로 금융 데이터를 분석하고 개인화 서비스를 제공합니다.",
                },
                {
                    "source": "placeholder",
                    "url": "https://www.dunamu.com",
                    "title": "두나무 - AI 금융 플랫폼",
                    "snippet": "AI 기반 자산관리, 거래 시스템, 리스크 관리",
                    "content": "두나무는 업비트를 운영하며 AI 기술을 활용한 금융 서비스를 제공하는 핀테크 기업입니다.",
                },
            ]
            search_results_raw = docs[:limit]

        docs = _dedupe_by_url(docs)[:limit * 2]

        # LLM Structured Output으로 정확한 추출 (from single file)
        candidates: List[Dict[str, Any]] = []
        
        if cfg.use_structured_output and LANGCHAIN_AVAILABLE and PYDANTIC_AVAILABLE:
            try:
                llm = ChatOpenAI(model=cfg.model, temperature=cfg.temperature)
                structured_llm = llm.with_structured_output(StartupSearchResult)
                
                sys_prompt = SYSTEM_PROMPT.format(limit=limit)
                user_prompt = USER_QUERY_TMPL.format(
                    query=query,
                    results=json.dumps(search_results_raw, ensure_ascii=False)
                )
                
                result: StartupSearchResult = structured_llm.invoke([
                    SystemMessage(content=sys_prompt),
                    HumanMessage(content=user_prompt)
                ])
                
                # 중복 제거
                uniq = {}
                for item in result.items:
                    key = item.name.strip().lower()
                    if key in uniq:
                        continue
                    uniq[key] = True
                    
                    # URL 찾기 (검색 결과에서)
                    item_url = ""
                    for d in docs:
                        if item.name.lower() in d.get("title", "").lower():
                            item_url = d.get("url", "")
                            break
                    
                    candidates.append({
                        "name": item.name.strip(),
                        "company": item.name.strip(),  # 호환성
                        "domain": item.domain.strip(),
                        "description": item.description.strip(),
                        "url": item_url,
                        "score": 0.95,  # structured output은 높은 신뢰도
                    })
                
                candidates = candidates[:limit]
                
            except Exception as e:
                print(f"⚠️ Structured output 실패, 기본 방식 사용: {e}")
                # 폴백: 기존 방식
                candidates = _fallback_candidates_extraction(docs, limit)
        else:
            # 기존 방식
            candidates = _fallback_candidates_extraction(docs, limit)

        state["startup_search"] = {
            "query": query,
            "candidates": candidates,
            "docs": docs,
            "ts": int(time.time()),
        }
        
        # 실행 결과 출력
        print("\n" + "="*80)
        print("✅ [1/6] 스타트업 검색 완료")
        print("="*80)
        print(f"📊 발견된 스타트업: {len(candidates)}개")
        for idx, c in enumerate(candidates[:3], 1):
            print(f"   {idx}. {c.get('name', 'N/A')} - {c.get('domain', 'N/A')}")
            print(f"      {c.get('description', '')[:60]}...")
        if len(candidates) > 3:
            print(f"   ... 외 {len(candidates)-3}개")
        print("="*80 + "\n")
        
        return state

    return node

def _fallback_candidates_extraction(docs: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """기존 방식의 후보 추출 (폴백용)"""
    candidates: List[Dict[str, Any]] = []
    for d in docs:
        company = _extract_company_from_title(d.get("title", ""))
        score = _score(d.get("title", ""), d.get("snippet", ""))
        candidates.append({
            "name": company,
            "company": company,
            "title": d.get("title"),
            "url": d.get("url"),
            "snippet": d.get("snippet"),
            "score": round(float(score), 3),
            "domain": "",
            "description": d.get("snippet", "")[:200]
        })
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:limit]

# ----------------------------- Graph Builder -----------------------------

def build_startup_search_graph(config: Optional[StartupSearchConfig] = None):
    cfg = config or StartupSearchConfig()
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph가 설치되어 있지 않습니다. `pip install langgraph` 후 다시 시도하세요.")
    g = StateGraph(StartupSearchState)  # 상태 타입 힌트
    g.add_node("startup_search", _startup_search_node_factory(cfg))
    g.add_edge(START, "startup_search")
    g.add_edge("startup_search", END)
    return g.compile()

# ----------------------------- Helper (Direct Invoke) -----------------------------

def run_startup_search(state: Dict[str, Any], config: Optional[StartupSearchConfig] = None) -> Dict[str, Any]:
    """그래프를 빌드하고 단일 노드를 실행하는 헬퍼."""
    app = build_startup_search_graph(config)
    return app.invoke(state)


# ----------------------------- Output Formatting -----------------------------

def print_startup_search_results(result: Dict[str, Any]):
    """검색 결과를 보기 좋게 출력"""
    search = result.get("startup_search", {})
    candidates = search.get("candidates", [])
    
    print("\n" + "=" * 80)
    print("🔍 스타트업 검색 결과")
    print("=" * 80)
    print(f"\n검색 쿼리: {search.get('query', 'N/A')}")
    print(f"총 발견: {len(candidates)}개 스타트업")
    print("\n" + "-" * 80)
    
    for idx, candidate in enumerate(candidates, 1):
        print(f"\n[{idx}] {candidate.get('name', candidate.get('company', 'Unknown'))}")
        if candidate.get('domain'):
            print(f"    📂 도메인: {candidate['domain']}")
        if candidate.get('description'):
            print(f"    💡 설명: {candidate['description']}")
        if candidate.get('url'):
            print(f"    🔗 URL: {candidate['url']}")
        print(f"    ⭐ 점수: {candidate.get('score', 0):.2f}")
    
    print("\n" + "=" * 80)


# ----------------------------- CLI Test -----------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Fintech Startup Finder (LangGraph)")
    parser.add_argument("--segment", default="fintech ai", help="검색 세그먼트")
    parser.add_argument("--region", default="Korea", help="지역 (Korea|Global)")
    parser.add_argument("--limit", type=int, default=10, help="검색할 스타트업 수")
    parser.add_argument("--no-structured", action="store_true", help="Structured output 비활성화")
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔍 한국 AI 핀테크 스타트업 검색")
    print("=" * 80)
    print(f"\n설정:")
    print(f"  - 세그먼트: {args.segment}")
    print(f"  - 지역: {args.region}")
    print(f"  - 최대 개수: {args.limit}")
    print(f"  - Structured Output: {not args.no_structured}")
    
    initial: StartupSearchState = {
        "segment": args.segment,
        "region": args.region,
        "limit": args.limit,
        "language": "ko"
    }
    
    config = StartupSearchConfig(
        use_structured_output=not args.no_structured,
        max_results=args.limit
    )
    
    try:
        final = run_startup_search(initial, config)
        print_startup_search_results(final)
        
        # JSON 출력 (옵션)
        if final.get("startup_search", {}).get("candidates"):
            output_file = "startup_search_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final.get("startup_search"), f, ensure_ascii=False, indent=2)
            print(f"\n💾 결과가 저장되었습니다: {output_file}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
