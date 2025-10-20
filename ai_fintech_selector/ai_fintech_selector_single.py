# ai_fintech_selector_single.py
"""
AI 핀테크 스타트업 선별 에이전트 (재탐색 기능 포함)
- 이전 에이전트의 결과를 받아서 가장 AI 핀테크 스타트업다운 회사 1개를 선정
- 적절한 후보가 없으면 탐색 에이전트로 돌아가서 새로운 리스트 요청
- 이미 검증한 회사는 제외 (중복 방지)

사용법: 
  # 파이프라인으로 연결
  from ai_fintech_selector_single import run_startup_selector
  state = run_startup_selector(previous_state, search_function)
"""

import os
import json
from typing import List, Dict, Any, Annotated, Callable, Optional
from dotenv import load_dotenv

from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch
from langchain_core.tools import tool


# ============================================================================
# 환경 설정
# ============================================================================
load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

MAX_RETRY_ROUNDS = 3  # 최대 재탐색 횟수


# ============================================================================
# 데이터 모델
# ============================================================================
class StartupCandidate(BaseModel):
    """후보 스타트업 정보"""
    name: str
    domain: str
    description: str
    
    model_config = ConfigDict(str_strip_whitespace=True)


class ValidationResult(BaseModel):
    """검증 결과"""
    is_fintech: bool = Field(description="핀테크 기업 여부")
    is_startup: bool = Field(description="스타트업 여부 (대기업/은행 제외)")
    uses_ai_core: bool = Field(description="AI를 핵심 기술로 활용하는지")
    growth_stage: str = Field(description="성장 단계: seed/early/growth/mature/unknown")
    funding_info: str = Field(description="투자 유치 정보 (Series A/B/C 등)")
    reasoning: str = Field(description="판단 근거")
    score: int = Field(description="종합 점수 (0-100)")
    
    model_config = ConfigDict(str_strip_whitespace=True)


class CompanyVerification(BaseModel):
    """회사 검증 정보"""
    company_name: str
    validation: ValidationResult
    additional_info: str = ""
    
    model_config = ConfigDict(str_strip_whitespace=True)


class SelectionState(BaseModel):
    """선별 에이전트 상태 - 다음 에이전트로 전달됨"""
    messages: Annotated[list, add_messages] = []
    
    # 입력 (이전 에이전트로부터)
    candidates: List[StartupCandidate] = []
    
    # 재탐색 관리
    excluded_names: List[str] = []  # 이미 검증한 회사명 (중복 방지)
    retry_count: int = 0  # 재탐색 횟수
    need_more_candidates: bool = False  # 재탐색 필요 여부
    
    # 중간 과정
    verification_results: List[CompanyVerification] = []
    all_verifications: List[CompanyVerification] = []  # 모든 라운드의 검증 결과
    
    # 출력 (다음 에이전트로 전달할 핵심 데이터)
    selected_name: str = ""
    selected_domain: str = ""
    selected_description: str = ""
    
    # 추가 메타데이터
    selection_reason: str = ""
    verification_score: int = 0
    
    # 메타
    current_step: str = "init"
    error: str = ""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


# ============================================================================
# Tools
# ============================================================================
_tavily = TavilySearch(
    max_results=10,
    api_key=TAVILY_API_KEY
)


@tool
def search_startup_info(company_name: str, query_focus: str) -> str:
    """
    스타트업 정보 검색
    
    Args:
        company_name: 회사명
        query_focus: 검색 초점 (funding/technology/business/validation)
    """
    focus_queries = {
        "funding": f"{company_name} 투자유치 시리즈 Series 펀딩",
        "technology": f"{company_name} AI 기술 머신러닝 알고리즘",
        "business": f"{company_name} 핀테크 금융 서비스 비즈니스모델",
        "validation": f"{company_name} 스타트업 대기업 은행 자회사"
    }
    
    query = focus_queries.get(query_focus, f"{company_name} {query_focus}")
    
    try:
        results = _tavily.invoke({"query": query})
        
        if isinstance(results, dict):
            results = results.get('results', [])
        
        if not results:
            return f"{company_name}에 대한 정보를 찾을 수 없습니다."
        
        summaries = []
        for i, r in enumerate(results[:5], 1):
            if isinstance(r, dict):
                title = r.get('title', 'No title')
                content = r.get('content', '')[:300]
                summaries.append(f"[{i}] {title}\n{content}")
        
        return "\n\n".join(summaries)
        
    except Exception as e:
        return f"검색 오류: {str(e)}"


# ============================================================================
# Prompts
# ============================================================================
VERIFICATION_SYSTEM_PROMPT = """당신은 스타트업 검증 전문가입니다.

**검증 기준:**

1. **핀테크 기업 (is_fintech)**
   - 금융 서비스/기술을 제공하는가?
   - 결제, 송금, 투자, 대출, 보험, 자산관리 등 금융 도메인
   - 단순 금융 소프트웨어 개발사는 제외

2. **스타트업 (is_startup)**
   ✅ 스타트업 특징:
   - 창업 초기~성장 단계 (통상 10년 이내)
   - 빠른 성장 추구 (10배 성장 목표)
   - VC/엔젤 투자 유치 이력
   - 혁신적 비즈니스 모델
   - M&A/IPO 목표
   
   ❌ 제외 대상:
   - 대기업 자회사/사업부
   - 은행/보험사/카드사의 자회사
   - 설립 15년 이상 성숙 기업
   - 안정적 수익 기반의 전통 중소기업

3. **AI 핵심 활용 (uses_ai_core)**
   - AI/ML이 제품/서비스의 핵심 차별화 요소인가?
   - 구체적 AI 기술: LLM, NLP, CV, 추천시스템, 이상탐지 등
   - 단순 "AI 도입" 수준은 불충분

4. **성장 단계 (growth_stage)**
   - seed: 아이디어/프로토타입
   - early: PMF 검증, Series A 전후
   - growth: 빠른 확장, Series B/C
   - mature: 안정적 성장, IPO 준비/완료
   - unknown: 정보 부족

5. **종합 점수 (0-100)**
   - 핀테크 적합도: 30점
   - 스타트업 특성: 40점 (가장 중요)
   - AI 핵심성: 30점

**합격 기준:**
- is_fintech = true
- is_startup = true  
- uses_ai_core = true
- score >= 60

이 모든 조건을 만족해야 합격입니다.
"""

VERIFICATION_USER_PROMPT = """다음 회사를 검증해주세요.

**회사 정보:**
- 이름: {company_name}
- 분야: {domain}
- 설명: {description}

**추가 검색 정보:**
{search_info}

위 정보를 바탕으로 ValidationResult 스키마에 맞춰 JSON으로 반환하세요.
특히 is_startup 판단에 신중을 기해주세요 (대기업/은행 계열사는 false).
"""

SELECTION_SYSTEM_PROMPT = """당신은 최종 선별 전문가입니다.

여러 후보 중 **가장 AI 핀테크 스타트업다운 회사 1개**를 선정하세요.

**합격 기준 (필수):**
- is_fintech = true
- is_startup = true
- uses_ai_core = true
- score >= 60

**합격자가 없는 경우:**
"NO_QUALIFIED_CANDIDATE"를 selected_name에 반환하세요.

**합격자가 있는 경우 선정 우선순위:**
1. 종합 점수 (score 높을수록 우선)
2. 성장 단계 (early/growth > seed/mature)
3. 투자 유치 이력 (Series A 이상 우대)

최종 선정 이유를 명확히 설명하세요.
"""

SELECTION_USER_PROMPT = """다음 검증 결과를 바탕으로 최적의 AI 핀테크 스타트업 1개를 선정하세요.

**검증 결과:**
{verification_results}

**출력 형식:**
{{
  "selected_name": "회사명 또는 NO_QUALIFIED_CANDIDATE",
  "selection_reason": "선정 이유 (3-5문장)"
}}

합격 기준을 만족하는 회사가 없으면 반드시 "NO_QUALIFIED_CANDIDATE"를 반환하세요.
JSON만 반환하세요.
"""


# ============================================================================
# LLM 초기화
# ============================================================================
_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)


# ============================================================================
# Nodes
# ============================================================================
def verification_node(state: SelectionState) -> SelectionState:
    """각 후보 회사 검증"""
    print(f"\n{'=' * 70}")
    print(f"🔍 검증 단계 (라운드 {state.retry_count + 1}): {len(state.candidates)}개 회사 분석")
    if state.excluded_names:
        print(f"   제외 목록: {', '.join(state.excluded_names)}")
    print("=" * 70)
    
    state.current_step = "verification"
    
    # Tools 바인딩
    llm_with_tools = _llm.bind_tools([search_startup_info])
    
    verification_results = []
    
    for idx, candidate in enumerate(state.candidates, 1):
        # 중복 체크
        if candidate.name in state.excluded_names:
            print(f"\n[{idx}/{len(state.candidates)}] {candidate.name} - ⏭️  이미 검증됨, 건너뜀")
            continue
        
        print(f"\n[{idx}/{len(state.candidates)}] {candidate.name} 검증 중...")
        
        # 검증한 회사 목록에 추가
        state.excluded_names.append(candidate.name)
        
        # 초기 정보로 1차 판단
        initial_info = f"회사: {candidate.name}\n분야: {candidate.domain}\n설명: {candidate.description}"
        
        # LLM에게 검색 필요 여부 판단 요청
        search_decision_prompt = f"""다음 회사 정보만으로 스타트업 여부를 확신할 수 있나요?

{initial_info}

불확실한 경우 (대기업 계열사 가능성, 설립연도 불명확 등) search_startup_info 도구를 사용하세요.
확실하면 "SUFFICIENT"라고만 답하세요.
"""
        
        messages = [
            SystemMessage(content="당신은 정보 충분성을 판단하는 전문가입니다."),
            HumanMessage(content=search_decision_prompt)
        ]
        
        response = llm_with_tools.invoke(messages)
        
        # 도구 호출 필요 시 실행
        additional_info = ""
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"   🔎 추가 검색 수행 중...")
            for tool_call in response.tool_calls[:2]:  # 최대 2번 검색
                if tool_call['name'] == 'search_startup_info':
                    args = tool_call['args']
                    search_result = search_startup_info.invoke(args)
                    additional_info += f"\n\n=== {args.get('query_focus', 'info')} 검색 ===\n{search_result}"
                    print(f"   ✅ {args.get('query_focus', 'info')} 정보 수집 완료")
        else:
            print(f"   ℹ️  기존 정보로 충분")
        
        # 최종 검증
        verification_prompt = VERIFICATION_USER_PROMPT.format(
            company_name=candidate.name,
            domain=candidate.domain,
            description=candidate.description,
            search_info=additional_info if additional_info else "추가 정보 없음"
        )
        
        try:
            structured_llm = _llm.with_structured_output(ValidationResult)
            validation = structured_llm.invoke([
                SystemMessage(content=VERIFICATION_SYSTEM_PROMPT),
                HumanMessage(content=verification_prompt)
            ])
            
            verification = CompanyVerification(
                company_name=candidate.name,
                validation=validation,
                additional_info=additional_info
            )
            
            verification_results.append(verification)
            state.all_verifications.append(verification)  # 전체 기록
            
            # 결과 출력
            is_qualified = (validation.is_fintech and 
                          validation.is_startup and 
                          validation.uses_ai_core and 
                          validation.score >= 60)
            
            status = "✅" if is_qualified else "⚠️"
            print(f"   {status} 점수: {validation.score}/100 | "
                  f"핀테크: {validation.is_fintech} | "
                  f"스타트업: {validation.is_startup} | "
                  f"AI핵심: {validation.uses_ai_core}")
            print(f"   📊 {validation.growth_stage} | {validation.funding_info}")
            if is_qualified:
                print(f"   🎯 합격!")
            
        except Exception as e:
            print(f"   ❌ 검증 실패: {e}")
            verification = CompanyVerification(
                company_name=candidate.name,
                validation=ValidationResult(
                    is_fintech=False,
                    is_startup=False,
                    uses_ai_core=False,
                    growth_stage="unknown",
                    funding_info="검증 실패",
                    reasoning=f"검증 오류: {str(e)}",
                    score=0
                )
            )
            verification_results.append(verification)
            state.all_verifications.append(verification)
    
    state.verification_results = verification_results
    return state


def decision_node(state: SelectionState) -> SelectionState:
    """선정 또는 재탐색 결정"""
    print(f"\n{'=' * 70}")
    print("🎯 선정 단계")
    print("=" * 70)
    
    state.current_step = "decision"
    
    if not state.verification_results:
        print("❌ 검증 결과가 없습니다.")
        state.error = "검증 결과 없음"
        return state
    
    # 검증 결과 요약
    results_summary = []
    for vr in state.verification_results:
        v = vr.validation
        summary = f"""
회사: {vr.company_name}
- 핀테크: {v.is_fintech} | 스타트업: {v.is_startup} | AI핵심: {v.uses_ai_core}
- 성장단계: {v.growth_stage} | 투자: {v.funding_info}
- 점수: {v.score}/100
- 근거: {v.reasoning}
"""
        results_summary.append(summary.strip())
    
    selection_prompt = f"""다음 검증 결과를 바탕으로 최적의 AI 핀테크 스타트업 1개를 선정하세요.

**검증 결과:**
{chr(10).join(results_summary)}

**합격 기준 (필수):**
- is_fintech = true
- is_startup = true
- uses_ai_core = true
- score >= 60

**합격자가 없는 경우:**
"NO_QUALIFIED_CANDIDATE"를 selected_name에 반환하세요.

**출력 형식 (반드시 JSON만):**
{{
  "selected_name": "회사명 또는 NO_QUALIFIED_CANDIDATE",
  "selection_reason": "선정 이유"
}}
"""
    
    try:
        response = _llm.invoke([
            SystemMessage(content=SELECTION_SYSTEM_PROMPT),
            HumanMessage(content=selection_prompt)
        ])
        
        # 응답 내용 확인
        content = response.content.strip()
        
        # JSON 추출 (코드 블록 제거)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # JSON 파싱
        result = json.loads(content)
        selected_name = result.get("selected_name", "")
        selection_reason = result.get("selection_reason", "")
        
        # 합격자 없음
        if selected_name == "NO_QUALIFIED_CANDIDATE":
            print(f"\n⚠️ 합격 기준을 만족하는 스타트업이 없습니다.")
            print(f"   사유: {selection_reason}")
            
            # 재탐색 필요 여부 결정
            if state.retry_count < MAX_RETRY_ROUNDS:
                state.need_more_candidates = True
                state.retry_count += 1
                print(f"\n🔄 재탐색 {state.retry_count}/{MAX_RETRY_ROUNDS}회차 시작")
            else:
                print(f"\n❌ 최대 재탐색 횟수({MAX_RETRY_ROUNDS}회) 도달. 선정 실패.")
                state.error = f"최대 재탐색 후에도 적합한 스타트업 없음"
            
            return state
        
        # 선정된 회사 찾기
        selected_candidate = None
        for candidate in state.candidates:
            if candidate.name == selected_name:
                selected_candidate = candidate
                break
        
        if selected_candidate:
            state.selected_name = selected_candidate.name
            state.selected_domain = selected_candidate.domain
            state.selected_description = selected_candidate.description
            state.selection_reason = selection_reason
            
            # 검증 점수 저장
            for vr in state.verification_results:
                if vr.company_name == selected_name:
                    state.verification_score = vr.validation.score
                    break
            
            state.need_more_candidates = False
            
            print(f"\n✅ 선정 완료: {state.selected_name}")
            print(f"   점수: {state.verification_score}/100")
            print(f"\n📝 선정 이유:\n{state.selection_reason}")
        else:
            print(f"⚠️ 선정된 회사를 찾을 수 없습니다: {selected_name}")
            state.error = f"선정된 회사 찾기 실패: {selected_name}"
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 실패: {e}")
        print(f"   LLM 응답: {content[:200]}...")
        state.error = f"JSON 파싱 오류: {str(e)}"
    except Exception as e:
        print(f"❌ 선정 실패: {e}")
        state.error = f"선정 오류: {str(e)}"
    
    return state


def output_node(state: SelectionState) -> SelectionState:
    """결과 출력"""
    print(f"\n{'=' * 70}")
    print("📊 최종 결과")
    print("=" * 70)
    
    state.current_step = "output"
    
    if state.error:
        print(f"\n❌ 오류: {state.error}")
        print(f"\n📋 전체 검증 이력 ({len(state.all_verifications)}개 회사):")
        for vr in state.all_verifications:
            v = vr.validation
            print(f"   - {vr.company_name}: {v.score}점 "
                  f"(핀테크:{v.is_fintech}, 스타트업:{v.is_startup}, AI:{v.uses_ai_core})")
        return state
    
    if not state.selected_name:
        print("\n❌ 선정된 스타트업이 없습니다.")
        return state
    
    # 결과 출력
    output = {
        "selected_startup": {
            "name": state.selected_name,
            "domain": state.selected_domain,
            "description": state.selected_description
        },
        "selection_reason": state.selection_reason,
        "verification_score": state.verification_score,
        "total_rounds": state.retry_count + 1,
        "total_verified": len(state.all_verifications),
        "all_verifications_summary": [
            {
                "name": vr.company_name,
                "score": vr.validation.score,
                "is_qualified": (vr.validation.is_fintech and 
                               vr.validation.is_startup and 
                               vr.validation.uses_ai_core and
                               vr.validation.score >= 60)
            }
            for vr in state.all_verifications
        ]
    }
    
    print(json.dumps(output, ensure_ascii=False, indent=2))
    
    # 파일 저장
    output_file = "selected_startup.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n💾 결과가 '{output_file}'에 저장되었습니다.")
    
    print(f"\n🎊 총 {state.retry_count + 1}라운드, {len(state.all_verifications)}개 회사 검증 완료!")
    
    return state


# ============================================================================
# 조건부 엣지: 재탐색 여부 결정
# ============================================================================
def should_retry(state: SelectionState) -> str:
    """재탐색 필요 여부 판단"""
    if state.need_more_candidates and state.retry_count <= MAX_RETRY_ROUNDS:
        return "need_more"
    else:
        return "done"


# ============================================================================
# Graph
# ============================================================================
def build_graph():
    """그래프 빌드"""
    workflow = StateGraph(SelectionState)
    
    workflow.add_node("verification", verification_node)
    workflow.add_node("decision", decision_node)
    workflow.add_node("output", output_node)
    
    workflow.set_entry_point("verification")
    workflow.add_edge("verification", "decision")
    
    # 조건부 엣지: 재탐색 필요시 END로 (외부에서 새 후보 받아서 재실행)
    workflow.add_conditional_edges(
        "decision",
        should_retry,
        {
            "need_more": END,  # 재탐색 필요 → 외부로 나가서 새 후보 받음
            "done": "output"   # 선정 완료 또는 최종 실패
        }
    )
    workflow.add_edge("output", END)
    
    return workflow.compile(checkpointer=MemorySaver())


# ============================================================================
# 실행 함수
# ============================================================================
def run_startup_selector(
    initial_candidates: List[Dict[str, str]],
    search_function: Callable[[List[str]], List[Dict[str, str]]]
) -> SelectionState:
    """스타트업 선별 실행"""
    app = build_graph()
    config = {"configurable": {"thread_id": "startup_selector_session"}}
    
    # 초기 상태
    startup_candidates = [StartupCandidate(**c) for c in initial_candidates]
    state = SelectionState(candidates=startup_candidates)
    
    # 재탐색 루프
    while True:
        # 그래프 실행 - 결과를 SelectionState로 변환
        result = app.invoke(state, config)
        
        # LangGraph 반환값을 SelectionState로 변환
        if isinstance(result, dict):
            # 딕셔너리를 SelectionState로 변환
            state = SelectionState(**{k: v for k, v in result.items() if k in SelectionState.model_fields})
        else:
            state = result
        
        # 선정 완료 또는 최종 실패
        if not state.need_more_candidates:
            break
        
        # 재탐색 필요
        print(f"\n{'=' * 70}")
        print(f"🔄 재탐색 시작 (제외 목록: {len(state.excluded_names)}개)")
        print("=" * 70)
        
        try:
            # 새로운 후보 가져오기
            new_candidates_dict = search_function(state.excluded_names)
            new_candidates = [StartupCandidate(**c) for c in new_candidates_dict]
            
            if not new_candidates:
                print("❌ 더 이상 새로운 후보를 찾을 수 없습니다.")
                state.error = "재탐색 실패: 새로운 후보 없음"
                break
            
            print(f"✅ 새로운 후보 {len(new_candidates)}개 발견")
            
            # 상태 업데이트
            state.candidates = new_candidates
            state.verification_results = []
            state.need_more_candidates = False
            
        except Exception as e:
            print(f"❌ 재탐색 실패: {e}")
            state.error = f"재탐색 오류: {str(e)}"
            break
    
    return state


# ============================================================================
# 메인 (테스트용)
# ============================================================================
def main():
    """테스트 실행"""
    
    # 모의 검색 함수 (실제로는 ai_fintech_search_single.py의 함수 사용)
    def mock_search_function(excluded_names: List[str]) -> List[Dict[str, str]]:
        """모의 재탐색 함수"""
        all_candidates = [
            # 1차 후보
            {"name": "토스", "domain": "Fintech/종합금융", "description": "간편송금 및 금융 슈퍼앱"},
            {"name": "카카오뱅크", "domain": "Fintech/은행", "description": "대기업 계열 인터넷은행"},
            {"name": "KB국민은행", "domain": "Fintech/은행", "description": "전통 은행"},
            # 2차 후보 (재탐색)
            {"name": "크레파스", "domain": "Fintech/신용평가", "description": "AI 기반 대안신용평가 스타트업"},
            {"name": "핀다", "domain": "Fintech/대출중개", "description": "AI 대출 비교 플랫폼"},
            # 3차 후보
            {"name": "뱅크샐러드", "domain": "Fintech/자산관리", "description": "개인 자산관리 앱. AI 기반 소비 분석"},
        ]
        
        # 제외 목록에 없는 후보만 반환
        return [c for c in all_candidates if c["name"] not in excluded_names][:3]
    
    # 초기 후보 (대기업/은행 포함 - 의도적으로 부적격)
    test_candidates = [
        {"name": "토스", "domain": "Fintech/종합금융", "description": "간편송금 및 금융 슈퍼앱"},
        {"name": "카카오뱅크", "domain": "Fintech/은행", "description": "대기업 계열 인터넷은행"},
        {"name": "KB국민은행", "domain": "Fintech/은행", "description": "전통 은행"}
    ]
    
    print("=" * 70)
    print("🚀 AI 핀테크 스타트업 선별 에이전트 (재탐색 기능 테스트)")
    print("=" * 70)
    
    result = run_startup_selector(test_candidates, mock_search_function)
    
    print(f"\n{'=' * 70}")
    print("✅ 선별 완료!")
    print("=" * 70)
    
    if result.selected_name:
        print(f"\n🏆 최종 선정: {result.selected_name}")
        print(f"📍 분야: {result.selected_domain}")
        print(f"📝 설명: {result.selected_description}")
        print(f"💯 점수: {result.verification_score}/100")
        print(f"🔄 총 라운드: {result.retry_count + 1}회")
        print(f"📊 검증한 회사: {len(result.all_verifications)}개")
    else:
        print(f"\n❌ 선정 실패: {result.error}")
        print(f"🔄 총 라운드: {result.retry_count + 1}회")
        print(f"📊 검증한 회사: {len(result.all_verifications)}개")


if __name__ == "__main__":
    main()