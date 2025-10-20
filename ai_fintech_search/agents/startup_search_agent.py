# ai_fintech_search/agents/startup_search_agent.py
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from ai_fintech_search.models import SearchState, StartupSearchResult, StartupHit
from ai_fintech_search.prompts.startup_search import SYSTEM_PROMPT, USER_QUERY_TMPL
from ai_fintech_search.tools.web_search import web_search
from ai_fintech_search.utils.normalize import normalize_root_url
from ai_fintech_search import config

# ✅ .env 키가 준비되었는지 런타임에서 체크
config.ensure_required_env()

# LLM (OPENAI_API_KEY는 langchain_openai가 환경변수에서 자동 사용)
_llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0.1)

def _startup_search_node(state: SearchState) -> SearchState:
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
        # Global이어도 한국 기업 우선 검색
        base_terms += [
            "Korean fintech AI companies",
            "South Korea fintech artificial intelligence"
        ]

    query = " OR ".join(base_terms)
    search_results = web_search(query)

    sys = SYSTEM_PROMPT.format(limit=state.limit)
    user = USER_QUERY_TMPL.format(query=query, results=search_results)

    structured = _llm.with_structured_output(StartupSearchResult)
    out: StartupSearchResult = structured.invoke([SystemMessage(content=sys),
                                                  HumanMessage(content=user)])

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
    state.results = cleaned[: state.limit]
    return state

def build_graph():
    graph = StateGraph(SearchState)
    graph.add_node("startup_search", _startup_search_node)
    graph.set_entry_point("startup_search")
    graph.set_finish_point("startup_search")
    return graph.compile(checkpointer=MemorySaver())

def run_startup_search(region: str = "Global", limit: int = 12, language: str = "ko") -> List[StartupHit]:
    app = build_graph()
    init = SearchState(region=region, limit=limit, language=language)
    config = {"configurable": {"thread_id": "startup_search_session"}}
    out = app.invoke(init, config)
    return out["results"]