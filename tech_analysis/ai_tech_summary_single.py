"""
AI 핀테크 기술 요약 에이전트 (LangGraph State 기반)
- 스타트업 검색 → 기술 요약을 LangGraph로 연결
- State를 통해 에이전트 간 데이터 전달
- 다음 에이전트가 추가되면 company_details를 바로 사용 가능

사용법: python ai_tech_summary_single.py --limit 5
"""

import os
import argparse
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

from pydantic import BaseModel, ConfigDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


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

    model_config = ConfigDict(str_strip_whitespace=True)


class StartupSearchResult(BaseModel):
    """LLM 구조화 출력용"""
    items: List[StartupHit] = []


class Company(BaseModel):
    """회사 기본 정보"""
    name: str
    domain: str
    description: str

    model_config = ConfigDict(str_strip_whitespace=True)


class Product(BaseModel):
    """제품/서비스 정보"""
    name: str
    description: str
    strengths: List[str]
    limitations: List[str]

    model_config = ConfigDict(str_strip_whitespace=True)


class CompanyDetail(BaseModel):
    """회사 상세 정보 (회사 + 제품들)"""
    company: Company
    products: List[Product]

    model_config = ConfigDict(str_strip_whitespace=True)


# ============================================================================
# LangGraph State 정의
# ============================================================================
class AgentState(BaseModel):
    """전체 파이프라인 상태"""
    # 입력 파라미터
    region: str = "Korea"
    limit: int = 5
    
    # 1단계: 스타트업 검색 결과
    startups: List[StartupHit] = []
    
    # 2단계: 기술 요약 결과 (다음 에이전트로 전달될 데이터)
    company_details: List[CompanyDetail] = []
    
    # 추가 메타데이터
    current_step: str = "init"
    error: str = ""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


# ============================================================================
# 도구 (Tools)
# ============================================================================
_tavily = TavilySearch(
    max_results=30,
    api_key=TAVILY_API_KEY
)

def web_search(query: str) -> List[Dict[str, Any]]:
    """웹 검색"""
    try:
        result = _tavily.invoke({"query": query})
        if isinstance(result, dict):
            return result.get('results', [])
        elif isinstance(result, list):
            return result
        else:
            return []
    except Exception as e:
        print(f"   ⚠️  검색 오류: {e}")
        return []


# ============================================================================
# 프롬프트
# ============================================================================
STARTUP_SEARCH_SYSTEM_PROMPT = """이 프롬프트는 사용되지 않습니다. 
선별 에이전트에서 받은 데이터를 직접 사용합니다."""

STARTUP_SEARCH_USER_QUERY_TMPL = """이 프롬프트는 사용되지 않습니다.
선별 에이전트에서 받은 데이터를 직접 사용합니다."""

TECH_SUMMARY_SYSTEM_PROMPT = """당신은 핀테크 원천 기술 분석 전문가입니다.

**분석 목표: 구체적인 AI 원천 기술 파악**

핵심 원칙:
1. **구체적인 기술만 추출**: "AI 활용" 같은 일반적 표현 금지
2. **검색 결과 기반**: 명시된 내용만 사용, 추측 금지
3. **기술 스택 우선**: 알고리즘, 모델, 프레임워크 등 구체적 기술 정보
4. **정보 부족 시 빈 리스트**: 장점이나 한계점 정보가 없으면 빈 리스트 [] 반환

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
  "company": {{"name": "", "domain": "", "description": ""}},
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
- 개요: {company_description}

=== 수집된 기술 정보 ===
{search_results}
========================

**분석 지침:**
1. 구체적인 AI/ML 알고리즘, 모델, 프레임워크를 찾으세요
2. "AI 활용", "머신러닝 기반" 같은 일반적 표현은 피하세요
3. 검색 결과에 명시된 기술만 사용하세요
4. 장점/한계점 정보가 없으면 빈 리스트 []로 반환하세요

JSON으로만 반환하세요.
"""


# ============================================================================
# LLM 초기화
# ============================================================================
_search_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)
_summary_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)


# ============================================================================
# 검색 노드는 제거됨 - 선별 에이전트에서 받은 데이터를 직접 사용
# ============================================================================


# ============================================================================
# 노드 1: 기술 요약
# ============================================================================
def collect_company_tech_info(company_name: str, company_domain: str) -> str:
    """회사에 대한 기술 정보를 웹에서 수집"""
    collected_info = []
    
    search_queries = [
        f"{company_name} 기술 스택 architecture AI",
        f"{company_name} AI 모델 알고리즘 머신러닝",
        f"{company_name} 개발 기술 블로그 tech",
        f"{company_name} engineering blog",
        f"{company_name} 특허 논문 patent"
    ]
    
    print(f"   🔍 기술 정보 검색 중...")
    
    seen_urls = set()
    collected_count = 0
    
    for query in search_queries[:4]:
        results = web_search(query)
        
        if not isinstance(results, list) or len(results) == 0:
            continue
        
        for result in results[:6]:
            if not isinstance(result, dict):
                continue
                
            title = result.get('title', '').lower()
            content = result.get('content', '')
            url = result.get('url', '')
            
            if url in seen_urls:
                continue
            
            tech_keywords = [
                'ai', 'ml', 'machine learning', 'deep learning', 'nlp', 
                'algorithm', 'model', '알고리즘', '모델', '머신러닝', 
                '딥러닝', '인공지능', 'tensorflow', 'pytorch', 'bert', 
                'gpt', 'transformer', 'lstm', 'xgboost', '기술', 'tech'
            ]
            
            has_company = company_name.lower() in title or company_name.lower() in content.lower()
            has_tech = any(keyword in title or keyword in content.lower() for keyword in tech_keywords)
            
            if has_company and has_tech and content and len(content) > 100:
                seen_urls.add(url)
                collected_info.append(
                    f"[출처: {result.get('title', 'Unknown')}]\n{content}\n"
                )
                collected_count += 1
                
                if collected_count >= 6:
                    break
        
        if collected_count >= 6:
            break
    
    if not collected_info:
        return f"{company_name}에 대한 기술 정보를 찾을 수 없습니다."
    
    print(f"   ✅ {collected_count}개 출처에서 정보 수집")
    return "\n\n".join(collected_info)


def tech_summary_node(state: AgentState) -> AgentState:
    """기술 요약 노드"""
    print(f"\n{'=' * 70}")
    print(f"🗜️ 1단계: 기술 요약 (총 {len(state.startups)}개 회사)")
    print("=" * 70)
    
    state.current_step = "tech_summary"
    
    if not state.startups:
        state.error = "검색된 스타트업이 없습니다"
        print(f"❌ {state.error}")
        return state
    
    company_details = []
    
    for startup in state.startups:
        print(f"\n🔬 [{startup.name}] 원천 기술 분석 시작...")
        
        # 회사 정보 수집
        search_results = collect_company_tech_info(startup.name, startup.domain)
        
        if "기술 정보를 찾을 수 없습니다" in search_results:
            print(f"   ⚠️  검색 결과 없음")
            company_details.append(CompanyDetail(
                company=Company(
                    name=startup.name,
                    domain=startup.domain,
                    description=startup.description
                ),
                products=[]
            ))
            continue
        
        # LLM 분석
        sys = TECH_SUMMARY_SYSTEM_PROMPT
        user = TECH_SUMMARY_USER_QUERY_TMPL.format(
            company_name=startup.name,
            company_domain=startup.domain,
            company_description=startup.description,
            search_results=search_results
        )
        
        try:
            structured = _summary_llm.with_structured_output(CompanyDetail)
            detail: CompanyDetail = structured.invoke([
                SystemMessage(content=sys),
                HumanMessage(content=user)
            ])
            
            if detail.products:
                print(f"   ✅ 분석 완료: 제품 {len(detail.products)}개")
                for prod in detail.products:
                    print(f"      - {prod.name}: {prod.description[:70]}...")
            else:
                print(f"   ⚠️  제품 정보 없음")
            
            company_details.append(detail)
            
        except Exception as e:
            print(f"   ❌ LLM 분석 실패: {e}")
            company_details.append(CompanyDetail(
                company=Company(
                    name=startup.name,
                    domain=startup.domain,
                    description=startup.description
                ),
                products=[]
            ))
    
    state.company_details = company_details
    return state


# ============================================================================
# 노드 2: 결과 출력 (다음 에이전트가 추가되면 이 노드 대신 다음 에이전트로)
# ============================================================================
def output_node(state: AgentState) -> AgentState:
    """결과 출력 노드 (나중에 다음 에이전트 노드로 대체될 예정)"""
    print(f"\n{'=' * 70}")
    print("📊 최종 결과")
    print("=" * 70)
    
    state.current_step = "output"
    
    # 통계
    total_products = sum(len(d.products) for d in state.company_details)
    companies_with_products = sum(1 for d in state.company_details if d.products)
    
    print(f"\n✅ 분석 완료!")
    print(f"   - 총 회사: {len(state.company_details)}개")
    print(f"   - 제품 정보 확인: {companies_with_products}개")
    print(f"   - 총 제품/서비스: {total_products}개")
    
    # JSON 출력
    output = {
        "total_companies": len(state.company_details),
        "companies": [detail.model_dump() for detail in state.company_details]
    }
    
    print(f"\n{json.dumps(output, ensure_ascii=False, indent=2)}")
    
    # 파일 저장
    output_file = "tech_summary_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n💾 결과가 '{output_file}'에 저장되었습니다.")
    
    return state


# ============================================================================
# 그래프 빌드
# ============================================================================
def build_graph():
    """LangGraph 빌드 - 검색 노드 없이 tech_summary부터 시작"""
    workflow = StateGraph(AgentState)
    
    # 노드 추가 (검색 노드 제거!)
    workflow.add_node("tech_summary", tech_summary_node)
    workflow.add_node("output", output_node)
    # 나중에 추가될 노드들:
    # workflow.add_node("market_evaluation", market_evaluation_node)
    # workflow.add_node("investment_decision", investment_decision_node)
    
    # 엣지 정의 - tech_summary부터 시작!
    workflow.set_entry_point("tech_summary")
    workflow.add_edge("tech_summary", "output")
    # 나중에 추가될 엣지들:
    # workflow.add_edge("tech_summary", "market_evaluation")
    # workflow.add_edge("market_evaluation", "investment_decision")
    workflow.add_edge("output", END)
    
    return workflow.compile(checkpointer=MemorySaver())


# ============================================================================
# 메인 실행
# ============================================================================
def main():
    p = argparse.ArgumentParser(description="AI Fintech Tech Summary Agent - LangGraph")
    p.add_argument("--region", default="Korea", help="Global | Korea")
    p.add_argument("--limit", type=int, default=5, help="분석할 스타트업 수")
    args = p.parse_args()

    print("=" * 70)
    print("🚀 AI 핀테크 원천 기술 분석 파이프라인 (LangGraph)")
    print("=" * 70)
    
    # 그래프 빌드
    app = build_graph()
    
    # 초기 상태
    initial_state = AgentState(
        region=args.region,
        limit=args.limit
    )
    
    # 그래프 실행
    config = {"configurable": {"thread_id": "tech_analysis_pipeline"}}
    final_state = app.invoke(initial_state, config)
    
    print(f"\n{'=' * 70}")
    print("✅ 파이프라인 완료!")
    print(f"최종 상태: {final_state.current_step}")
    if final_state.error:
        print(f"에러: {final_state.error}")
    print("=" * 70)
    
    # 다음 에이전트로 전달할 데이터 (company_details)
    print(f"\n💡 다음 에이전트에게 전달될 데이터:")
    print(f"   - company_details: {len(final_state.company_details)}개 회사 상세 정보")
    print(f"   - 각 회사는 Company + List[Product] 구조")
    
    return final_state


if __name__ == "__main__":
    main()