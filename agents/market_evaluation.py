"""
Module: agents/market_evaluation.py (Agentic RAG 통합 버전)
Purpose: 시장성 평가 에이전트 (Query Rewrite + Relevance Check + Web Fallback)

입력 State 요구:
state = {
  "tech_summary": {"company": "Zest AI", "products": [...]},  # CompanyDetail 지원
  "segment": "fintech ai fraud detection",
  "report_dir": "path/to/pdfs",  # PDF 리포트 폴더
  "market_store": [...],  # 선택: 사전 축적 RAG
  "startup_search": {...}  # 선택: 백업 문서
}

출력 병합:
state["market_eval"] = {
  "company_name": "Zest AI",
  "market_size": {TAM, SAM, SOM, CAGR, ...},
  "growth_potential": {...},
  "competitive_landscape": {...},
  "regulatory_environment": {...},
  "overall_score": 8.5,
  "investment_recommendation": "강력추천",
  # 레거시 호환
  "market_score": 79,
  "risk_score": 73,
}
"""
from __future__ import annotations

import os
import json
import warnings
from typing import Any, Dict, List, Optional, TypedDict, Literal
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# .env 로드
here = Path(__file__).resolve().parent
candidates = [
    here / ".env",
    here.parent / ".env",
    here.parents[1] / ".env",
    Path.cwd() / ".env",
]
env_used = None
for p in candidates:
    if p.exists():
        load_dotenv(p)
        env_used = str(p)
        break

if not env_used:
    load_dotenv()
    env_used = "default"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# LangGraph
try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

# LangChain
try:
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
    from langchain_openai import ChatOpenAI
    LANGCHAIN_CHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_CHAIN_AVAILABLE = False

# Tavily
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except Exception as e:
    TAVILY_AVAILABLE = False
    print(f"⚠️ Tavily import 실패: {e}")

# GroundednessChecker
try:
    from langchain_teddynote.evaluator import GroundednessChecker
    GROUNDEDNESS_AVAILABLE = True
except Exception:
    GROUNDEDNESS_AVAILABLE = False

# RAG 라이브러리
RAG_AVAILABLE = True
missing_libs = []

try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception as e:
    RAG_AVAILABLE = False
    missing_libs.append(f"PyPDFLoader (pip install langchain-community)")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception as e:
    RAG_AVAILABLE = False
    missing_libs.append(f"RecursiveCharacterTextSplitter (pip install langchain-text-splitters)")

try:
    from langchain_openai import OpenAIEmbeddings
    EMBEDDINGS_AVAILABLE = True
except Exception:
    EMBEDDINGS_AVAILABLE = False
    missing_libs.append(f"OpenAIEmbeddings (pip install langchain-openai)")

try:
    from langchain_community.vectorstores import FAISS
except Exception as e:
    RAG_AVAILABLE = False
    missing_libs.append(f"FAISS (pip install faiss-cpu)")

if not RAG_AVAILABLE:
    print("⚠️ RAG 라이브러리가 설치되지 않았습니다.")
    print("누락된 라이브러리:", ", ".join(missing_libs))
else:
    print("✅ RAG 라이브러리 로드 완료")

# Pydantic 스키마
try:
    from pydantic import BaseModel, Field, ConfigDict
    
    class Company(BaseModel):
        name: str
        desription: str  # 오타 유지 (tech_summary.py와 호환)
        domain: str
    
    class Product(BaseModel):
        name: str
        description: str
        strengths: List[str]
        limitations: List[str]
    
    class CompanyDetail(BaseModel):
        company: Company
        products: List[Product]
    
    class MarketSize(BaseModel):
        """시장 규모 정보"""
        TAM: Optional[float] = Field(None, description="Total Addressable Market (단위: 억원)")
        SAM: Optional[float] = Field(None, description="Serviceable Available Market (단위: 억원)")
        SOM: Optional[float] = Field(None, description="Serviceable Obtainable Market (단위: 억원)")
        currency: str = Field("KRW", description="통화 단위")
        year: int = Field(2024, description="기준 연도")
        source: Optional[str] = Field(None, description="데이터 출처")
        confidence: float = Field(0.0, ge=0, le=1, description="신뢰도 (0-1)")

    class GrowthPotential(BaseModel):
        """성장 가능성 평가"""
        CAGR: Optional[float] = Field(None, description="연평균 성장률 (%)")
        growth_drivers: List[str] = Field(default_factory=list, description="성장 동인")
        market_trends: List[str] = Field(default_factory=list, description="시장 트렌드")
        adoption_stage: str = Field("", description="기술 채택 단계")
        timeframe: str = Field("2024-2028", description="예측 기간")
        source: Optional[str] = Field(None, description="데이터 출처")

    class CompetitiveLandscape(BaseModel):
        """경쟁 환경 분석"""
        intensity: str = Field("", description="경쟁 강도 (낮음/중간/높음/매우높음)")
        key_players: List[str] = Field(default_factory=list, description="주요 경쟁사")
        entry_barriers: List[str] = Field(default_factory=list, description="진입 장벽")
        differentiation: str = Field("", description="차별화 포인트")
        market_position: str = Field("", description="예상 시장 포지션")

    class RegulatoryEnvironment(BaseModel):
        """규제 환경"""
        risk_level: str = Field("", description="규제 리스크 (낮음/중간/높음)")
        key_regulations: List[str] = Field(default_factory=list, description="주요 규제")
        compliance_cost: str = Field("", description="준수 비용")
        licensing_required: bool = Field(False, description="라이선스 필요 여부")
        recent_changes: List[str] = Field(default_factory=list, description="최근 규제 변경")

    class MarketEvaluation(BaseModel):
        """시장성 종합 평가 결과"""
        company_name: str
        evaluation_date: datetime = Field(default_factory=datetime.now)
        
        market_size: MarketSize
        growth_potential: GrowthPotential
        competitive_landscape: CompetitiveLandscape
        regulatory_environment: RegulatoryEnvironment
        
        overall_score: float = Field(0.0, ge=0, le=10, description="종합 점수 (0-10)")
        investment_recommendation: str = Field("", description="투자 추천 (강력추천/추천/보류/비추천)")
        key_insights: List[str] = Field(default_factory=list, description="핵심 인사이트")
        risk_factors: List[str] = Field(default_factory=list, description="리스크 요인")
        opportunities: List[str] = Field(default_factory=list, description="기회 요인")
        
        data_sources: List[str] = Field(default_factory=list, description="데이터 출처")
        confidence_level: float = Field(0.0, ge=0, le=1, description="평가 신뢰도")
        analyst_notes: str = Field("", description="분석가 노트")
    
    PYDANTIC_AVAILABLE = True
except Exception:
    PYDANTIC_AVAILABLE = False

# ----------------------------- State & Config -----------------------------

class MarketEvalOutput(TypedDict, total=False):
    # 신규 필드 (Agentic RAG)
    company_name: str
    market_size: Dict[str, Any]
    growth_potential: Dict[str, Any]
    competitive_landscape: Dict[str, Any]
    regulatory_environment: Dict[str, Any]
    overall_score: float
    investment_recommendation: str
    key_insights: List[str]
    risk_factors: List[str]
    opportunities: List[str]
    confidence_level: float
    # 레거시 호환
    company: str
    market_summary: str
    competitors: List[str]
    risk_summary: str
    market_score: float
    risk_score: float
    sources: List[Dict[str, str]]

class MarketEvalState(TypedDict, total=False):
    tech_summary: Dict[str, Any]
    segment: str
    market_store: List[Dict[str, str]]
    startup_search: Dict[str, Any]
    report_dir: str
    company_detail: Dict[str, Any]
    market_eval: MarketEvalOutput

# 내부 Agentic State
class AgenticMarketState(BaseModel):
    """Agentic RAG 내부 상태"""
    company_detail: Optional[CompanyDetail] = None
    original_query: str = ""
    rewritten_query: str = ""
    retrieved_docs_pdf: str = ""
    retrieved_docs_web: str = ""
    relevance: str = ""  # "yes" or "no"
    evaluation: Optional[MarketEvaluation] = None
    current_step: str = "init"
    error: str = ""
    model_config = ConfigDict(arbitrary_types_allowed=True)

@dataclass
class MarketEvalConfig:
    model: str = "gpt-4o-mini"
    model_advanced: str = "gpt-4o"
    temperature: float = 0.2
    use_agentic_rag: bool = True  # Agentic RAG 사용
    report_dir: str = None
    top_k: int = 5
    
    def __post_init__(self):
        if self.report_dir is None:
            current_file = Path(__file__).resolve()
            self.report_dir = str(current_file.parent.parent / "data")

# ----------------------------- PDF Retriever -----------------------------

class SimplePDFRetriever:
    """간단한 PDF 검색기 (FAISS)"""
    def __init__(self, pdf_dir: str, top_k: int = 5):
        self.pdf_dir = pdf_dir
        self.top_k = top_k
        self.vectorstore = None
        self.retriever = None
        self._initialize()
    
    def _initialize(self):
        """PDF 로드 및 벡터스토어 초기화"""
        if not RAG_AVAILABLE or not EMBEDDINGS_AVAILABLE:
            return
        
        pdf_paths = list(Path(self.pdf_dir).glob("*.pdf")) + list(Path(self.pdf_dir).glob("*.PDF"))
        if not pdf_paths:
            warnings.warn(f"No PDFs in {self.pdf_dir}")
            return
        
        all_docs = []
        for pdf_path in pdf_paths[:4]:  # 최대 4개
            try:
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load()
                all_docs.extend(docs[:50])  # 페이지 제한
            except Exception as e:
                warnings.warn(f"Failed to load {pdf_path}: {e}")
        
        if all_docs:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(all_docs)
            
            self.vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
    
    def retrieve(self, query: str) -> str:
        """문서 검색"""
        if not self.retriever:
            return ""
        try:
            docs = self.retriever.invoke(query)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            return ""

# ----------------------------- Prompts -----------------------------

QUERY_REWRITE_PROMPT = PromptTemplate(
    template="""당신은 핀테크 시장 분석 전문가입니다.
주어진 질문을 벡터 검색에 최적화된 형태로 재작성하세요.

# 재작성 가이드라인
1. 핵심 키워드 강조 (예: "핀테크", "시장 규모", "성장률", "CAGR", "경쟁사")
2. 구체적인 도메인 명시 (예: "신용평가", "결제", "대출", "자산관리")
3. 정량적 데이터 요청 명확화 (예: "TAM/SAM/SOM", "연평균 성장률")
4. 불필요한 수식어 제거
5. 원래 언어(한국어) 유지

# 원본 질문
{question}

# 재작성된 질문 (질문만 출력, 설명 없이)
""",
    input_variables=["question"]
)

MARKET_EVAL_SYSTEM_PROMPT = """당신은 20년 경력의 핀테크 시장 분석 전문가입니다.

# 당신의 역할
스타트업의 시장 잠재력을 평가하여 투자 의사결정을 지원합니다.

# 평가 기준
1. **시장 규모 (Market Size)**
   - TAM/SAM/SOM (단위: 억원)

2. **성장 가능성 (Growth Potential)**
   - CAGR, 성장 동인, 시장 트렌드

3. **경쟁 환경 (Competitive Landscape)**
   - 경쟁 강도, 주요 경쟁사, 진입 장벽

4. **규제 환경 (Regulatory Environment)**
   - 규제 리스크, 주요 규제

# 평가 원칙
- 데이터 기반 분석 (검색된 리포트 우선)
- 보수적 추정
- 출처 명시
- 한국 시장 중심

# 투자 추천 기준
- 강력추천 (8.0~10.0): 대규모 시장, 고성장, 낮은 경쟁
- 추천 (6.0~7.9): 중규모 시장, 중고성장
- 보류 (4.0~5.9): 시장 불확실성
- 비추천 (0~3.9): 작은 시장, 저성장

{format_instructions}
"""

MARKET_EVAL_USER_PROMPT = """# 회사 정보
- 이름: {company_name}
- 도메인: {domain}
- 설명: {description}

# 제품/서비스
{products_info}

# 검색된 시장 데이터
{retrieved_data}

# 요청사항
위 정보를 바탕으로 종합적인 시장성 평가를 수행하세요.
"""

# ----------------------------- Agentic Nodes -----------------------------

def _query_rewrite_node(cfg: MarketEvalConfig):
    def node(state: AgenticMarketState) -> AgenticMarketState:
        """질문 재작성"""
        print("\n[1/5] 🔄 Query Rewrite...")
        try:
            if not LANGCHAIN_CHAIN_AVAILABLE:
                state.rewritten_query = state.original_query
                return state
            
            llm = ChatOpenAI(model=cfg.model, temperature=0)
            chain = QUERY_REWRITE_PROMPT | llm | StrOutputParser()
            state.rewritten_query = chain.invoke({"question": state.original_query}).strip()
            print(f"   ✨ {state.rewritten_query[:80]}...")
        except Exception as e:
            state.error = f"Query rewrite failed: {e}"
            state.rewritten_query = state.original_query
        return state
    return node

def _retrieve_pdf_node(cfg: MarketEvalConfig):
    retriever = SimplePDFRetriever(cfg.report_dir, cfg.top_k) if cfg.report_dir else None
    
    def node(state: AgenticMarketState) -> AgenticMarketState:
        """PDF 검색"""
        print("\n[2/5] 📚 Retrieving from PDF...")
        if not retriever or not retriever.retriever:
            print("   ⚠️ PDF retriever unavailable")
            state.retrieved_docs_pdf = ""
            return state
        
        try:
            query = state.rewritten_query or state.original_query
            state.retrieved_docs_pdf = retriever.retrieve(query)
            print(f"   📄 Retrieved {len(state.retrieved_docs_pdf)} chars")
        except Exception as e:
            state.error = f"PDF retrieval failed: {e}"
            state.retrieved_docs_pdf = ""
        return state
    return node

def _relevance_check_node(cfg: MarketEvalConfig):
    def node(state: AgenticMarketState) -> AgenticMarketState:
        """관련성 체크"""
        print("\n[3/5] ✅ Checking relevance...")
        if not state.retrieved_docs_pdf:
            state.relevance = "no"
            print("   ⚠️ No PDF docs - will use web search")
            return state
        
        if not GROUNDEDNESS_AVAILABLE or not LANGCHAIN_CHAIN_AVAILABLE:
            state.relevance = "yes"  # 체크 불가 시 기본 yes
            return state
        
        try:
            checker = GroundednessChecker(
                llm=ChatOpenAI(model=cfg.model, temperature=0),
                target="question-retrieval"
            ).create()
            
            response = checker.invoke({
                "question": state.rewritten_query,
                "context": state.retrieved_docs_pdf[:3000]
            })
            state.relevance = response.score  # "yes" or "no"
            print(f"   {'✅ RELEVANT' if state.relevance == 'yes' else '⚠️ NOT RELEVANT'}")
        except Exception as e:
            state.relevance = "yes"  # 에러 시 기본값
            print(f"   ⚠️ Check failed: {e}")
        return state
    return node

def _web_search_node(cfg: MarketEvalConfig):
    def node(state: AgenticMarketState) -> AgenticMarketState:
        """웹 검색 (PDF 실패 시)"""
        print("\n[4/5] 🌐 Web Search...")
        if not TAVILY_AVAILABLE:
            print("   ⚠️ Tavily unavailable")
            state.retrieved_docs_web = ""
            return state
        
        try:
            company = state.company_detail.company
            query = f"{company.name} {company.domain} 시장 규모 성장률 핀테크 한국"
            
            # TavilyClient 사용
            client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            res = client.search(query=query, max_results=5)
            state.retrieved_docs_web = "\n\n".join([r.get("content", "") for r in res.get("results", [])])
            
            print(f"   🔍 Found {len(state.retrieved_docs_web)} chars")
        except Exception as e:
            state.error = f"Web search failed: {e}"
            state.retrieved_docs_web = ""
        return state
    return node

def _generate_evaluation_node(cfg: MarketEvalConfig):
    def node(state: AgenticMarketState) -> AgenticMarketState:
        """평가 생성"""
        print("\n[5/5] 🎯 Generating evaluation...")
        if not PYDANTIC_AVAILABLE or not LANGCHAIN_CHAIN_AVAILABLE:
            print("   ⚠️ Dependencies unavailable")
            return state
        
        try:
            company = state.company_detail.company
            products = state.company_detail.products
            
            # 데이터 선택 (PDF 우선)
            if state.relevance == "yes" and state.retrieved_docs_pdf:
                retrieved_data = state.retrieved_docs_pdf
                print("   📚 Using PDF reports")
            elif state.retrieved_docs_web:
                retrieved_data = state.retrieved_docs_web
                print("   🌐 Using web search")
            else:
                retrieved_data = "검색 데이터 없음"
                print("   ⚠️ No data - using general knowledge")
            
            # 제품 정보 포맷
            products_info = "\n".join([
                f"- {p.name}: {p.description}\n  강점: {', '.join(p.strengths[:3])}"
                for p in products
            ])
            
            # LLM 체인
            parser = PydanticOutputParser(pydantic_object=MarketEvaluation)
            prompt = ChatPromptTemplate.from_messages([
                ("system", MARKET_EVAL_SYSTEM_PROMPT),
                ("user", MARKET_EVAL_USER_PROMPT)
            ])
            llm = ChatOpenAI(model=cfg.model_advanced, temperature=cfg.temperature)
            chain = prompt | llm | parser
            
            state.evaluation = chain.invoke({
                "format_instructions": parser.get_format_instructions(),
                "company_name": company.name,
                "domain": company.domain,
                "description": company.desription,  # 오타 필드 사용
                "products_info": products_info,
                "retrieved_data": retrieved_data[:4000]
            })
            
            print(f"   ✅ Score: {state.evaluation.overall_score}/10, {state.evaluation.investment_recommendation}")
        except Exception as e:
            state.error = f"Evaluation failed: {e}"
            print(f"   ❌ Error: {e}")
        return state
    return node

def _route_after_relevance(state: AgenticMarketState) -> Literal["web_search", "generate"]:
    """관련성 체크 후 분기"""
    return "generate" if state.relevance == "yes" else "web_search"

# ----------------------------- Agentic Graph -----------------------------

def _build_agentic_graph(cfg: MarketEvalConfig):
    """Agentic RAG 그래프 생성"""
    workflow = StateGraph(AgenticMarketState)
    
    workflow.add_node("query_rewrite", _query_rewrite_node(cfg))
    workflow.add_node("retrieve_pdf", _retrieve_pdf_node(cfg))
    workflow.add_node("relevance_check", _relevance_check_node(cfg))
    workflow.add_node("web_search", _web_search_node(cfg))
    workflow.add_node("generate", _generate_evaluation_node(cfg))
    
    workflow.add_edge("query_rewrite", "retrieve_pdf")
    workflow.add_edge("retrieve_pdf", "relevance_check")
    workflow.add_conditional_edges(
        "relevance_check",
        _route_after_relevance,
        {"web_search": "web_search", "generate": "generate"}
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    
    workflow.set_entry_point("query_rewrite")
    return workflow.compile()

# ----------------------------- Node Factory -----------------------------

def _market_eval_node_factory(cfg: MarketEvalConfig):
    def node(state: MarketEvalState) -> MarketEvalState:
        company_detail_data = state.get("company_detail") or (state.get("tech_summary", {}) or {}).get("company_detail")
        report_dir = state.get("report_dir") or cfg.report_dir
        
        # Agentic RAG 모드
        if cfg.use_agentic_rag and company_detail_data and report_dir and PYDANTIC_AVAILABLE:
            try:
                detail = CompanyDetail(**company_detail_data)
                
                # 원본 쿼리 생성
                original_query = f"""
                {detail.company.name} 회사의 {detail.company.domain} 분야에 대한 시장성 평가
                - 시장 규모 (TAM/SAM/SOM)
                - 연평균 성장률 (CAGR)
                - 경쟁 환경
                - 규제 현황
                - 핀테크 {detail.company.desription}
                """
                
                # Agentic 그래프 실행
                cfg.report_dir = report_dir
                agentic_graph = _build_agentic_graph(cfg)
                agentic_state = AgenticMarketState(
                    company_detail=detail,
                    original_query=original_query.strip()
                )
                result = agentic_graph.invoke(agentic_state)
                
                # result가 dictionary일 경우 처리
                eval_obj = None
                if isinstance(result, dict):
                    eval_obj = result.get('evaluation')
                elif hasattr(result, 'evaluation'):
                    eval_obj = result.evaluation
                
                if eval_obj:
                    # 경쟁사 목록에서 대상 회사 제외
                    import re
                    def normalize_name(name):
                        """회사 이름 정규화 (괄호, 공백 제거)"""
                        name = re.sub(r'\([^)]*\)', '', name).strip()
                        name = re.sub(r'\s+', ' ', name).strip()
                        return name.lower()
                    
                    company_normalized = normalize_name(eval_obj.company_name)
                    filtered_competitors = [
                        comp for comp in eval_obj.competitive_landscape.key_players
                        if normalize_name(comp) != company_normalized
                    ][:5]
                    
                    # Pydantic 모델을 dict로 변환하는 헬퍼 함수
                    def to_dict(obj):
                        if hasattr(obj, 'model_dump'):
                            return obj.model_dump()
                        elif hasattr(obj, 'dict'):
                            return obj.dict()
                        elif isinstance(obj, dict):
                            return obj
                        else:
                            return obj
                    
                    # 레거시 필드 추가
                    output = {
                        "company_name": eval_obj.company_name,
                        "market_size": to_dict(eval_obj.market_size),
                        "growth_potential": to_dict(eval_obj.growth_potential),
                        "competitive_landscape": to_dict(eval_obj.competitive_landscape),
                        "regulatory_environment": to_dict(eval_obj.regulatory_environment),
                        "overall_score": eval_obj.overall_score,
                        "investment_recommendation": eval_obj.investment_recommendation,
                        "key_insights": eval_obj.key_insights,
                        "risk_factors": eval_obj.risk_factors,
                        "opportunities": eval_obj.opportunities,
                        "confidence_level": eval_obj.confidence_level,
                        # 레거시 호환
                        "company": eval_obj.company_name,
                        "market_summary": f"TAM: {eval_obj.market_size.TAM or 'N/A'}, CAGR: {eval_obj.growth_potential.CAGR or 'N/A'}%",
                        "competitors": filtered_competitors,  # 대상 회사 제외된 경쟁사 목록
                        "risk_summary": ", ".join(eval_obj.risk_factors[:3]),
                        "market_score": eval_obj.overall_score * 10,  # 0-10 -> 0-100
                        "risk_score": max(0, 100 - len(eval_obj.risk_factors) * 10),
                        "sources": [{"title": src, "url": ""} for src in eval_obj.data_sources[:5]]
                    }
                    
                    state["market_eval"] = output
                    
                    # 실행 결과 출력 (Agentic RAG 모드)
                    print("\n" + "="*80)
                    print("✅ [3/6] 시장 평가 완료 (Agentic RAG)")
                    print("="*80)
                    print(f"🏢 회사: {output.get('company_name', 'N/A')}")
                    print(f"📊 종합 점수: {output.get('overall_score', 0)}/10")
                    print(f"💡 추천: {output.get('investment_recommendation', 'N/A')}")
                    print(f"🎯 신뢰도: {output.get('confidence_level', 0):.1%}")
                    if output.get('key_insights'):
                        print(f"💡 핵심 인사이트:")
                        for idx, insight in enumerate(output['key_insights'][:2], 1):
                            print(f"   {idx}. {insight[:60]}...")
                    print("="*80 + "\n")
                    
                    return state
            except Exception as e:
                warnings.warn(f"Agentic RAG 실패, 레거시 모드로 전환: {e}")
        
        # 레거시 모드 (기존 로직)
        company = (state.get("tech_summary") or {}).get("company", "(unknown)")
        state["market_eval"] = {
            "company": company,
            "market_summary": "핀테크 시장은 디지털 결제 증가로 성장 중",
            "competitors": ["Competitor A", "Competitor B"],
            "risk_summary": "규제 리스크 존재",
            "market_score": 75.0,
            "risk_score": 70.0,
            "sources": []
        }
        
        # 실행 결과 출력 (레거시 모드)
        print("\n" + "="*80)
        print("✅ [3/6] 시장 평가 완료 (Legacy)")
        print("="*80)
        print(f"🏢 회사: {company}")
        print(f"📊 시장 점수: {state['market_eval']['market_score']}/100")
        print(f"⚠️ 리스크 점수: {state['market_eval']['risk_score']}/100")
        print(f"🏆 경쟁사: {', '.join(state['market_eval']['competitors'])}")
        print("="*80 + "\n")
        
        return state
    return node

# ----------------------------- Graph Builder -----------------------------

def build_market_eval_graph(config: Optional[MarketEvalConfig] = None):
    cfg = config or MarketEvalConfig()
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph가 설치되어 있지 않습니다.")
    g = StateGraph(MarketEvalState)
    g.add_node("market_eval", _market_eval_node_factory(cfg))
    g.add_edge(START, "market_eval")
    g.add_edge("market_eval", END)
    return g.compile()

def run_market_eval(state: Dict[str, Any], config: Optional[MarketEvalConfig] = None) -> Dict[str, Any]:
    app = build_market_eval_graph(config)
    return app.invoke(state)

# ----------------------------- CLI Test -----------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Market Evaluation Agent (Agentic RAG)")
    parser.add_argument("--company", default="업스테이지", help="회사명")
    parser.add_argument("--segment", default="fintech ai", help="세그먼트")
    parser.add_argument("--report-dir", default=None, help="PDF 리포트 폴더")
    parser.add_argument("--no-agentic", action="store_true", help="Agentic RAG 비활성화")
    args = parser.parse_args()
    
    print("=" * 80)
    print("📊 시장 평가 에이전트 테스트 (Agentic RAG)")
    print("=" * 80)
    
    if args.report_dir:
        report_dir = args.report_dir
    else:
        current_file = Path(__file__).resolve()
        report_dir = str(current_file.parent.parent / "data")
    
    print(f"\n설정:")
    print(f"  - 회사: {args.company}")
    print(f"  - PDF 경로: {report_dir}")
    print(f"  - Agentic RAG: {not args.no_agentic}")
    
    dummy: MarketEvalState = {
        "tech_summary": {
            "company": args.company,
            "company_detail": {
                "company": {
                    "name": args.company,
                    "desription": "AI 기반 신용평가 및 사기 탐지 솔루션",  # 오타 유지
                    "domain": "FinTech/Credit Scoring"
                },
                "products": [
                    {
                        "name": "AI Credit Scoring",
                        "description": "머신러닝 기반 신용평가 모델",
                        "strengths": ["실시간 처리", "높은 정확도"],
                        "limitations": ["설명가능성"]
                    }
                ]
            }
        },
        "segment": args.segment,
        "report_dir": report_dir
    }
    
    config = MarketEvalConfig(
        use_agentic_rag=not args.no_agentic,
        report_dir=report_dir
    )
    
    try:
        final = run_market_eval(dummy, config)
        
        print("\n" + "=" * 80)
        print("📊 시장 평가 결과")
        print("=" * 80)
        
        market = final.get("market_eval", {})
        print(f"\n회사: {market.get('company_name', market.get('company', 'N/A'))}")
        
        if "overall_score" in market:
            print(f"종합 점수: {market['overall_score']}/10")
            print(f"투자 추천: {market.get('investment_recommendation', 'N/A')}")
            print(f"신뢰도: {market.get('confidence_level', 0):.1%}")
            
            if market.get('market_size'):
                ms = market['market_size']
                print(f"\n시장 규모:")
                print(f"  TAM: {ms.get('TAM', 'N/A')}")
                print(f"  SAM: {ms.get('SAM', 'N/A')}")
            
            if market.get('key_insights'):
                print(f"\n핵심 인사이트:")
                for insight in market['key_insights'][:3]:
                    print(f"  - {insight}")
        else:
            print(f"시장 점수: {market.get('market_score', 0):.1f}")
            print(f"리스크 점수: {market.get('risk_score', 0):.1f}")
        
        # JSON 저장
        output_file = "market_eval_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(market, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 결과 저장: {output_file}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
