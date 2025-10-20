# ai_fintech_search_single.py
"""
AI 핀테크 스타트업 검색 에이전트 (단일 파일 버전)
사용법: python ai_fintech_search_single.py --region Korea --limit 10
"""

import os
import argparse
import json
from typing import List, Annotated
from urllib.parse import urlparse, urlunparse
from dotenv import load_dotenv

from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch


# ============================================================================
# 환경 설정
# ============================================================================
load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

def ensure_required_env():
    """필수 환경변수 확인"""
    required = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        lines = [f"- {k} 가 설정되지 않았습니다." for k in missing]
        msg = "다음 환경변수가 필요합니다:\n" + "\n".join(lines) + "\n(.env에 키를 설정하세요)"
        raise RuntimeError(msg)

ensure_required_env()


# ============================================================================
# 데이터 모델
# ============================================================================
class StartupHit(BaseModel):
    """스타트업 검색 결과"""
    name: str
    domain: str
    description: str

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )


class StartupSearchResult(BaseModel):
    """LLM 구조화 출력용"""
    items: List[StartupHit] = []


class SearchState(BaseModel):
    """에이전트 상태"""
    messages: Annotated[list, add_messages] = []
    region: str = "Global"
    limit: int = 12
    language: str = "ko"
    results: List[StartupHit] = []
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


# ============================================================================
# 유틸리티
# ============================================================================
def normalize_root_url(url: str) -> str:
    """URL 정규화"""
    try:
        u = urlparse(url)
        clean = u._replace(path="/", params="", query="", fragment="")
        return urlunparse(clean)
    except Exception:
        return url


# ============================================================================
# 도구 (Tools)
# ============================================================================
_tavily = TavilySearch(
    max_results=30,
    api_key=TAVILY_API_KEY
)

def web_search(query: str) -> list:
    """웹 검색"""
    return _tavily.invoke({"query": query})


# ============================================================================
# 프롬프트
# ============================================================================
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


# ============================================================================
# 에이전트
# ============================================================================
_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)


def _startup_search_node(state: SearchState) -> SearchState:
    """스타트업 검색 노드"""
    # 한국 스타트업에 집중
    base_terms = [
        "한국 AI 신용평가 스타트업",
        "국내 로보어드바이저 스타트업",
        "한국 AI 대출 핀테크",
        "국내 이상거래탐지 FDS AI",
        "한국 핀테크 스타트업 투자유치",
        "국내 금융 AI 스타트업 시리즈"
    ]
    
    if state.region.lower() == "korea":
        base_terms += [
            "크레파스 뱅크샐러드 토스",
            "핀테크 스타트업 AI 기술 활용",
            "금융 인공지능 스타트업 사례"
        ]
    else:
        base_terms += [
            "Korean fintech AI companies",
            "South Korea fintech artificial intelligence"
        ]

    query = " OR ".join(base_terms)
    search_results = web_search(query)

    sys = SYSTEM_PROMPT.format(limit=state.limit)
    user = USER_QUERY_TMPL.format(query=query, results=search_results)

    structured = _llm.with_structured_output(StartupSearchResult)
    out: StartupSearchResult = structured.invoke([
        SystemMessage(content=sys),
        HumanMessage(content=user)
    ])

    uniq = {}
    cleaned: List[StartupHit] = []
    for item in out.items:
        key = item.name.strip().lower()
        if key in uniq:
            continue
        uniq[key] = True
        cleaned.append(StartupHit(
            name=item.name.strip(),
            domain=item.domain.strip(),
            description=item.description.strip()
        ))
    state.results = cleaned[:state.limit]
    return state


def build_graph():
    """그래프 빌드"""
    graph = StateGraph(SearchState)
    graph.add_node("startup_search", _startup_search_node)
    graph.set_entry_point("startup_search")
    graph.set_finish_point("startup_search")
    return graph.compile(checkpointer=MemorySaver())


def run_startup_search(region: str = "Global", limit: int = 12, language: str = "ko") -> List[StartupHit]:
    """스타트업 검색 실행"""
    app = build_graph()
    init = SearchState(region=region, limit=limit, language=language)
    config = {"configurable": {"thread_id": "startup_search_session"}}
    out = app.invoke(init, config)
    return out["results"]


# ============================================================================
# 메인 실행
# ============================================================================
def main():
    p = argparse.ArgumentParser(description="AI Fintech Startup Finder (2025)")
    p.add_argument("--region", default="Korea", help="Global | Korea")
    p.add_argument("--limit", type=int, default=10, help="검색할 스타트업 수")
    args = p.parse_args()

    print(f"🔍 한국 AI 핀테크 스타트업 검색 중... (최대 {args.limit}개)\n")
    
    hits = run_startup_search(region=args.region, limit=args.limit, language="ko")
    
    output = {"items": [h.model_dump() for h in hits]}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    
    print(f"\n✅ 총 {len(hits)}개의 스타트업을 찾았습니다.")


if __name__ == "__main__":
    main()