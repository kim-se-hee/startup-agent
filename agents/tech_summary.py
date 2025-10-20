"""
Module: agents/tech_summary.py (LangGraph + 원천 기술 분석 통합 버전)
Purpose: "기술 요약" 에이전트 (원천 기술 상세 분석, LangGraph 기반)

특징:
- 구체적인 AI/ML 알고리즘, 모델, 프레임워크 식별
- CompanyDetail (회사 + 제품 리스트) 구조로 출력
- 웹 검색을 통한 기술 정보 수집
- Pydantic Structured Output 지원

입력 State:
state = {
  "startup_search": {
      "candidates": [{"name": "토스", "domain": "Fintech/결제", "description": "..."}],
      "docs": [...]
  },
  "target_company": "토스"   # (선택)
}

출력 State:
state["tech_summary"] = {
  "company": "토스",
  "summary": "...",
  "strengths": [...],
  "weaknesses": [...],
  "tech_score": 0~100,
  "sources": [...],
  # 신규: CompanyDetail 구조
  "company_detail": {
      "company": {"name": "", "domain": "", "description": ""},
      "products": [
          {"name": "", "description": "", "strengths": [], "limitations": []}
      ]
  }
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
    from langchain_core.messages import SystemMessage, HumanMessage
    LANGCHAIN_CHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_CHAIN_AVAILABLE = False

# Tavily (웹 검색)
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except Exception as e:
    TAVILY_AVAILABLE = False
    print(f"⚠️ Tavily import 실패: {e}")

# Pydantic (스키마)
try:
    from pydantic import BaseModel, Field, ConfigDict
    
    class Company(BaseModel):
        """회사 기본 정보"""
        name: str
        domain: str
        desription: str  # 오타 유지 (market.py 호환성)
        model_config = ConfigDict(str_strip_whitespace=True)
    
    class Product(BaseModel):
        """제품/서비스 정보"""
        name: str
        description: str
        strengths: List[str] = []
        limitations: List[str] = []
        model_config = ConfigDict(str_strip_whitespace=True)
    
    class CompanyDetail(BaseModel):
        """회사 상세 정보 (회사 + 제품들)"""
        company: Company
        products: List[Product]
        model_config = ConfigDict(str_strip_whitespace=True)
    
    PYDANTIC_AVAILABLE = True
except Exception:
    PYDANTIC_AVAILABLE = False
    Company = dict
    Product = dict
    CompanyDetail = dict


# ----------------------------- State & Config -----------------------------
class TechSummaryOutput(TypedDict, total=False):
    company: str
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    tech_score: float
    sources: List[Dict[str, str]]
    # 신규: CompanyDetail 구조
    company_detail: Dict[str, Any]
    products: List[Dict[str, Any]]

class TechSummaryState(TypedDict, total=False):
    startup_search: Dict[str, Any]
    target_company: str
    tech_summary: TechSummaryOutput

@dataclass
class TechSummaryConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0  # 원천 기술 분석은 정확성 중요
    include_chars: int = 1800
    use_structured_output: bool = True  # CompanyDetail structured output 사용
    collect_web_info: bool = True  # 추가 웹 검색 여부


# ----------------------------- Helpers -----------------------------
def _norm(s: str) -> str:
    return (s or "").strip()

def _normalize_company_name(name: str) -> str:
    """회사 이름 정규화 (괄호 제거 등)"""
    import re
    # 괄호와 내용 제거: "PFCT(피에프씨테크놀로지스)" -> "PFCT"
    name = re.sub(r'\([^)]*\)', '', name).strip()
    # 연속 공백 제거
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def _print_tech_summary_result(tech_summary: Dict[str, Any]):
    """기술 요약 결과 출력"""
    print("\n" + "="*80)
    print("✅ [2/6] 기술 요약 완료")
    print("="*80)
    print(f"🏢 회사: {tech_summary.get('company', 'N/A')}")
    print(f"📊 기술 점수: {tech_summary.get('tech_score', 0):.1f}/100")
    
    strengths = tech_summary.get('strengths', [])
    if strengths:
        print(f"💪 기술 강점:")
        for idx, s in enumerate(strengths[:3], 1):
            print(f"   {idx}. {s}")
    
    company_detail = tech_summary.get('company_detail', {})
    products = company_detail.get('products', [])
    if products:
        print(f"📦 제품/서비스: {len(products)}개")
        for idx, p in enumerate(products[:2], 1):
            print(f"   {idx}. {p.get('name', 'N/A')}: {p.get('description', '')[:50]}...")
    
    print("="*80 + "\n")

def _choose_company(state: TechSummaryState) -> tuple:
    """회사 정보 선택 (name, domain, description)"""
    if _norm(state.get("target_company", "")):
        company_name = _norm(state["target_company"])
        # startup_search에서 추가 정보 찾기
        cands = (state.get("startup_search", {}) or {}).get("candidates", [])
        for c in cands:
            if _norm(c.get("name", "")).lower() == company_name.lower():
                return (c.get("name"), c.get("domain", ""), c.get("description", ""))
        return (company_name, "", "")
    
    cands = (state.get("startup_search", {}) or {}).get("candidates", [])
    if not cands:
        return ("(unknown)", "", "")
    
    c = cands[0]
    return (
        _norm(c.get("name") or c.get("company") or c.get("title", "").split(" - ")[0]),
        _norm(c.get("domain", "")),
        _norm(c.get("description", ""))
    )

def _related_docs(state: TechSummaryState, company: str, max_docs: int = 8, max_chars: int = 2000) -> List[Dict[str, str]]:
    """기존 문서에서 관련 문서 필터링"""
    docs = (state.get("startup_search", {}) or {}).get("docs", [])
    company_l = company.lower()
    picked: List[Dict[str, str]] = []
    for d in docs:
        title = _norm(d.get("title", ""))
        url = _norm(d.get("url", ""))
        content = _norm(d.get("content", "")) or _norm(d.get("snippet", ""))
        if company_l in title.lower() or company_l in url.lower() or company_l in content.lower():
            picked.append({"title": title[:240], "url": url, "content": content[:max_chars]})
    if not picked:
        for d in docs[:3]:
            picked.append({
                "title": _norm(d.get("title", ""))[:240],
                "url": _norm(d.get("url", "")),
                "content": (_norm(d.get("content", "")) or _norm(d.get("snippet", "")))[:max_chars]
            })
    return picked[:max_docs]

def collect_company_tech_info(company_name: str, company_domain: str) -> str:
    """회사에 대한 기술 정보를 웹에서 수집 (from single file)"""
    if not TAVILY_AVAILABLE:
        return f"{company_name}에 대한 기술 정보를 찾을 수 없습니다."
    
    # 회사 이름 정규화 (괄호 제거)
    normalized_name = _normalize_company_name(company_name)
    print(f"🔍 기술 정보 검색: {normalized_name}")
    
    collected_info = []
    
    search_queries = [
        f"{normalized_name} 기술 스택 architecture AI",
        f"{normalized_name} AI 모델 알고리즘 머신러닝",
        f"{normalized_name} 개발 기술 블로그 tech",
        f"{normalized_name} engineering blog",
        f"{normalized_name} 특허 논문 patent"
    ]
    
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return f"{company_name}에 대한 기술 정보를 찾을 수 없습니다. (Tavily API Key 없음)"
    
    try:
        tavily_client = TavilyClient(api_key=tavily_key)
    except Exception as e:
        return f"{company_name}에 대한 기술 정보를 찾을 수 없습니다. (Tavily 초기화 실패: {e})"
    
    seen_urls = set()
    collected_count = 0
    
    for query in search_queries[:4]:
        try:
            response = tavily_client.search(query=query, max_results=6)
            results = response.get('results', [])
            
            if not isinstance(results, list):
                continue
            
            for result in results[:6]:
                if not isinstance(result, dict):
                    continue
                    
                title = result.get('title', '').lower()
                content = result.get('content', '')
                url = result.get('url', '')
                
                if url in seen_urls or not content or len(content) < 100:
                    continue
                
                tech_keywords = [
                    'ai', 'ml', 'machine learning', 'deep learning', 'nlp', 
                    'algorithm', 'model', '알고리즘', '모델', '머신러닝', 
                    '딥러닝', '인공지능', 'tensorflow', 'pytorch', 'bert', 
                    'gpt', 'transformer', 'lstm', 'xgboost', '기술', 'tech'
                ]
                
                has_company = normalized_name.lower() in title or normalized_name.lower() in content.lower()
                has_tech = any(keyword in title or keyword in content.lower() for keyword in tech_keywords)
                
                if has_company and has_tech:
                    seen_urls.add(url)
                    collected_info.append(
                        f"[출처: {result.get('title', 'Unknown')}]\n{content}\n"
                    )
                    collected_count += 1
                    
                    if collected_count >= 6:
                        break
            
            if collected_count >= 6:
                break
                
        except Exception:
            continue
    
    if not collected_info:
        return f"{company_name}에 대한 기술 정보를 찾을 수 없습니다."
    
    return "\n\n".join(collected_info)

def _offline_summarize() -> Dict[str, Any]:
    """오프라인 기본값"""
    return {
        "summary": (
            "지도학습과 이상탐지를 결합한 모델을 사용하며, 거래 로그/디바이스 신호 데이터를 활용. "
            "실시간 추론 파이프라인과 모니터링 체계를 보유하고, API 형태로 금융기관 워크플로우에 통합 가능."
        ),
        "strengths": ["실시간 처리 아키텍처", "데이터 피처 엔지니어링 자산", "클라우드 네이티브 운영"],
        "weaknesses": ["설명가능성/규제 적합성 과제", "데이터 편향 리스크"],
        "tech_score": 82,
    }


# ----------------------------- Prompts (from single file) -----------------------------

TECH_SUMMARY_SYSTEM_PROMPT = """당신은 핀테크 원천 기술 분석 전문가입니다.

**분석 목표: 구체적인 AI 원천 기술 파악**

핵심 원칙:
1. **구체적인 기술만 추출**: "AI 활용" 같은 일반적 표현 금지
2. **검색 결과 기반**: 명시된 내용만 사용, 추측 금지
3. **기술 스택 우선**: 알고리즘, 모델, 프레임워크 등 구체적 기술 정보
4. **강점/한계 필수**: 각 제품의 strengths와 limitations를 반드시 분석하여 추출

분석 작업:

1. **제품/서비스 식별**
   - 검색 결과에서 명확한 제품/서비스만 추출
   - 여러 제품이 있으면 각각 별도 객체로 생성

2. **원천 기술 설명 (description) - 가장 중요**
   다음 정보를 우선 순위대로 추출:
   
   a) **AI/ML 알고리즘**: 
      - 예: "Random Forest 기반 신용평가", "LSTM 시계열 예측", "Transformer 기반 NLP"
      - 예: "XGBoost 리스크 모델", "강화학습 추천 시스템"
      
   b) **구체적 모델**:
      - 예: "BERT 파인튜닝", "GPT 기반 챗봇", "CNN 문서 분석"
      
   c) **기술 프레임워크/라이브러리**:
      - 예: "TensorFlow", "PyTorch", "scikit-learn", "Keras"
      
   d) **데이터 처리 기술**:
      - 예: "비정형 데이터 처리", "실시간 스트림 처리", "대규모 분산 처리"

3. **기술적 강점 (strengths) - 필수**
   제품/기술의 장점을 추출:
   - 예: "실시간 처리 가능", "높은 정확도", "낮은 운영 비용"
   - 예: "대용량 데이터 처리", "빠른 응답 속도", "자동화된 의사결정"
   - 검색 결과에 명시된 내용만 사용
   - 최소 2-3개 이상 추출 시도

4. **기술적 한계 (limitations) - 필수**
   제품/기술의 제약사항 추출:
   - 예: "설명가능성 부족", "데이터 편향 가능성", "초기 구축 비용"
   - 예: "특정 도메인 제한", "학습 데이터 의존성", "정기적 재학습 필요"
   - 검색 결과에 명시된 내용만 사용
   - 최소 1-2개 이상 추출 시도
      
   e) **특화 기술**:
      - 예: "자연어처리(NLP)", "컴퓨터 비전(CV)", "추천 엔진", "이상 탐지"

   **작성 형식**:
   "[서비스명]: [구체적 AI 기술] 기반의 [금융 서비스]. [추가 기술 상세]"
   
   **좋은 예시**:
   - "AI 신용평가: Gradient Boosting(XGBoost) 알고리즘 기반 대안신용평가 모델. 비정형 데이터(통신, 결제 등) 학습"
   - "챗봇: BERT 파인튜닝한 금융 상담 NLP 모델. 의도 분류 및 엔티티 추출"
   
   **나쁜 예시**:
   - "AI 기술을 활용한 신용평가" ❌
   - "머신러닝 기반 서비스" ❌

   **정보 부족시**: "핀테크 [분야] 서비스 (구체적 기술 정보 부족)"

3. **장점 (strengths)**
   - 기술적 차별점, 성과, 정확도 개선 등
   - 구체적 수치 포함 (예: "정확도 15% 향상")
   - 최소 2개, 최대 5개
   - **정보 없으면: 빈 리스트 []**

4. **한계점 (limitations)**
   - 기술적 제약, 개선 과제
   - **정보 없으면: 빈 리스트 []**

출력 스키마 (JSON만):
{{
  "company": {{"name": "", "domain": "", "desription": ""}},
  "products": [
    {{
      "name": "",
      "description": "",
      "strengths": [],
      "limitations": []
    }}
  ]
}}
"""

TECH_SUMMARY_USER_QUERY_TMPL = """다음 회사의 원천 기술 정보를 분석하세요.

회사 정보:
- 이름: {company_name}
- 분야: {company_domain}  
- 개요: {company_desription}

=== 수집된 기술 정보 ===
{search_results}
========================

**분석 지침:**
1. 구체적인 AI/ML 알고리즘, 모델, 프레임워크를 찾으세요
2. "AI 활용", "머신러닝 기반" 같은 일반적 표현은 피하세요
3. 각 제품마다 **strengths(강점)와 limitations(한계)**를 반드시 추출하세요
   - 검색 결과에서 기술의 장점, 특징, 이점 → strengths
   - 검색 결과에서 제약사항, 한계, 리스크 → limitations
   - 정보가 부족하면 일반적인 AI 기술의 강점/한계 추론 가능
4. 검색 결과에 명시된 내용을 우선하되, 제품 특성상 명확한 강점/한계는 추론 가능

**필수 출력:**
- 각 제품의 strengths: 최소 2-3개
- 각 제품의 limitations: 최소 1-2개

JSON으로만 반환하세요.
"""


# ----------------------------- Node Factory -----------------------------
def _get_llm(model: str, temperature: float):
    if not LANGCHAIN_CHAIN_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        return ChatOpenAI(model=model, temperature=temperature)
    except Exception:
        return None

def _llm_json(prompt: str, llm) -> Dict[str, Any]:
    if not llm:
        return _offline_summarize()
    try:
        chain = ChatPromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
        txt = chain.invoke({"prompt": prompt})
        return json.loads(txt)
    except Exception:
        return _offline_summarize()

def _tech_summary_node_factory(cfg: TechSummaryConfig):
    def node(state: TechSummaryState) -> TechSummaryState:
        company_name, company_domain, company_desc = _choose_company(state)
        
        # Structured Output 방식 (from single file)
        if cfg.use_structured_output and LANGCHAIN_CHAIN_AVAILABLE and PYDANTIC_AVAILABLE:
            try:
                # 웹에서 기술 정보 추가 수집
                if cfg.collect_web_info:
                    search_results = collect_company_tech_info(company_name, company_domain)
                else:
                    # 기존 문서만 사용
                    docs = _related_docs(state, company_name, max_docs=8, max_chars=cfg.include_chars)
                    search_results = "\n\n".join([f"[{d['title']}]\n{d['content']}" for d in docs])
                
                if "기술 정보를 찾을 수 없습니다" in search_results:
                    # 기본값 반환
                    state["tech_summary"] = {
                        "company": company_name,
                        "summary": f"{company_name}의 기술 정보를 찾을 수 없습니다.",
                        "strengths": [],
                        "weaknesses": [],
                        "tech_score": 50,
                        "sources": [],
                        "company_detail": {
                            "company": {"name": company_name, "domain": company_domain, "desription": company_desc},  # 오타 필드명
                            "products": []
                        }
                    }
                    _print_tech_summary_result(state["tech_summary"])
                    return state
                
                # LLM Structured Output
                llm = ChatOpenAI(model=cfg.model, temperature=cfg.temperature)
                structured_llm = llm.with_structured_output(CompanyDetail)
                
                sys_prompt = TECH_SUMMARY_SYSTEM_PROMPT
                user_prompt = TECH_SUMMARY_USER_QUERY_TMPL.format(
                    company_name=company_name,
                    company_domain=company_domain,
                    company_desription=company_desc,  # 오타 필드명
                    search_results=search_results
                )
                
                detail: CompanyDetail = structured_llm.invoke([
                    SystemMessage(content=sys_prompt),
                    HumanMessage(content=user_prompt)
                ])
                
                # CompanyDetail을 레거시 형식으로 변환
                all_strengths = []
                all_limitations = []
                product_summaries = []
                
                for prod in detail.products:
                    product_summaries.append(f"{prod.name}: {prod.description}")
                    all_strengths.extend(prod.strengths)
                    all_limitations.extend(prod.limitations)
                
                summary_text = "\n".join(product_summaries) if product_summaries else company_desc
                
                # 기술 점수 계산 (제품 수와 기술 상세도 기반)
                tech_score = min(100, 60 + len(detail.products) * 10 + len(all_strengths) * 3)
                
                state["tech_summary"] = {
                    "company": company_name,
                    "summary": summary_text,
                    "strengths": all_strengths[:5],
                    "weaknesses": all_limitations[:5],
                    "tech_score": float(tech_score),
                    "sources": [],
                    # CompanyDetail 추가 (market_eval에서 사용)
                    "company_detail": {
                        "company": {
                            "name": detail.company.name,
                            "desription": detail.company.desription,  # 오타 필드명 (호환성)
                            "domain": detail.company.domain
                        },
                        "products": [
                            {
                                "name": p.name,
                                "description": p.description,
                                "strengths": p.strengths,
                                "limitations": p.limitations
                            } for p in detail.products
                        ]
                    },
                    "products": [p.model_dump() if hasattr(p, 'model_dump') else p for p in detail.products]
                }
                _print_tech_summary_result(state["tech_summary"])
                return state
                
            except Exception as e:
                print(f"⚠️ Structured output 실패, 기존 방식으로 전환: {e}")
        
        # 기존 방식 (폴백)
        docs = _related_docs(state, company_name, max_docs=8, max_chars=cfg.include_chars)
        llm = _get_llm(cfg.model, cfg.temperature)

        prompt = f"""
너는 핀테크 AI 기술 분석가다. 아래 JSON 문서를 근거로 **{company_name}** 의 기술을 요약하라.
근거 없는 추정은 금지하며, 불확실하면 언급하지 말라.

입력 문서(JSON): {json.dumps(docs, ensure_ascii=False)[:7000]}

아래 형식의 JSON만 반환:
{{
  "summary": "핵심 기술/아키텍처/데이터/성능 요약 (8~12문장)",
  "strengths": ["최대 5개"],
  "weaknesses": ["최대 5개"],
  "tech_score": 0-100
}}
"""
        data = _llm_json(prompt, llm)
        sources = [{"title": d.get("title", ""), "url": d.get("url", "")} for d in docs[:5]]

        state["tech_summary"] = {
            "company": company_name,
            "summary": data.get("summary", ""),
            "strengths": data.get("strengths", []),
            "weaknesses": data.get("weaknesses", []),
            "tech_score": float(data.get("tech_score", 0)),
            "sources": sources,
        }
        _print_tech_summary_result(state["tech_summary"])
        return state
    return node


# ----------------------------- Graph Builder -----------------------------
def build_tech_summary_graph(config: Optional[TechSummaryConfig] = None):
    cfg = config or TechSummaryConfig()
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph가 설치되어 있지 않습니다. `pip install langgraph` 후 다시 시도하세요.")
    g = StateGraph(TechSummaryState)
    g.add_node("tech_summary", _tech_summary_node_factory(cfg))
    g.add_edge(START, "tech_summary")
    g.add_edge("tech_summary", END)
    return g.compile()


# ----------------------------- Helper for Direct Run -----------------------------
def run_tech_summary(state: Dict[str, Any], config: Optional[TechSummaryConfig] = None) -> Dict[str, Any]:
    app = build_tech_summary_graph(config)
    return app.invoke(state)


# ----------------------------- Output Formatting -----------------------------

def print_tech_summary_results(result: Dict[str, Any]):
    """기술 요약 결과를 보기 좋게 출력"""
    tech = result.get("tech_summary", {})
    
    print("\n" + "=" * 80)
    print("🔬 기술 요약 결과")
    print("=" * 80)
    
    print(f"\n🏢 회사: {tech.get('company', 'N/A')}")
    print(f"📊 기술 점수: {tech.get('tech_score', 0):.1f} / 100")
    
    if tech.get('summary'):
        print(f"\n💡 기술 요약:")
        print(f"{tech['summary']}")
    
    if tech.get('strengths'):
        print(f"\n✅ 강점 ({len(tech['strengths'])}개):")
        for idx, strength in enumerate(tech['strengths'], 1):
            print(f"   {idx}. {strength}")
    
    if tech.get('weaknesses'):
        weakness_count = len(tech.get('weaknesses', []))
        print(f"\n⚠️  약점 ({weakness_count}개):")
        for idx, weakness in enumerate(tech['weaknesses'], 1):
            print(f"   {idx}. {weakness}")
    
    # CompanyDetail 구조 출력
    if tech.get('company_detail'):
        detail = tech['company_detail']
        products = detail.get('products', [])
        if products:
            print(f"\n📦 제품/서비스 ({len(products)}개):")
            for idx, prod in enumerate(products, 1):
                print(f"\n   [{idx}] {prod.get('name', 'N/A')}")
                print(f"       설명: {prod.get('description', '')[:100]}...")
                if prod.get('strengths'):
                    print(f"       강점: {', '.join(prod['strengths'][:3])}")
                if prod.get('limitations'):
                    print(f"       한계: {', '.join(prod['limitations'][:3])}")
    
    print("\n" + "=" * 80)


# ----------------------------- CLI Test -----------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tech Summary Agent (LangGraph)")
    parser.add_argument("--company", default="토스", help="분석할 회사명")
    parser.add_argument("--domain", default="Fintech/결제", help="회사 도메인")
    parser.add_argument("--no-web-search", action="store_true", help="웹 검색 비활성화")
    parser.add_argument("--no-structured", action="store_true", help="Structured output 비활성화")
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔬 기술 요약 에이전트 테스트")
    print("=" * 80)
    print(f"\n설정:")
    print(f"  - 회사: {args.company}")
    print(f"  - 도메인: {args.domain}")
    print(f"  - 웹 검색: {not args.no_web_search}")
    print(f"  - Structured Output: {not args.no_structured}")
    
    # 모의 startup_search 결과
    dummy = {
        "startup_search": {
            "candidates": [{
                "name": args.company,
                "domain": args.domain,
                "description": f"{args.company}는 AI 기술을 활용하는 핀테크 기업입니다."
            }],
            "docs": []
        },
        "target_company": args.company,
    }
    
    config = TechSummaryConfig(
        use_structured_output=not args.no_structured,
        collect_web_info=not args.no_web_search
    )
    
    try:
        final = run_tech_summary(dummy, config)
        print_tech_summary_results(final)
        
        # JSON 저장
        output_file = "tech_summary_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final.get("tech_summary"), f, ensure_ascii=False, indent=2)
        print(f"\n💾 결과가 저장되었습니다: {output_file}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()