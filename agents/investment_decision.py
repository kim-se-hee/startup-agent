"""
Module: agents/investment_decision.py (LangGraph + fintech 통합 버전)
Purpose: 투자 판단 에이전트 (가중치 기반 평가, LangGraph 기반)

입력 State 요구:
state = {
  "tech_summary": {"tech_score": 82, "summary": "..."},
  "market_eval": {"market_score": 79, "risk_score": 73, "market_summary": "..."},
  "competition": {"edge_score": 76, "differentiation_summary": "..."},
}

출력 병합:
state["investment"] = {
  "total_score": 0~5.0 (가중치 기반),
  "weights": {"ROI":0.2, "기술_경쟁력":0.2, "시장성":0.2, "경쟁_우위":0.15, "팀_역량":0.15, "리스크":0.1},
  "scores": {"ROI":4.0, "기술_경쟁력":5.0, ...},
  "breakdown": {...},  # 항목별 상세 정보
  "decision": "투자 적격 (Invest)|보류 (Hold)",
  "rationale": "요약 근거",
}
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional, TypedDict, List
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

# LangChain LLM
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI
    LANGCHAIN_CHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_CHAIN_AVAILABLE = False


# ----------------------------- State & Config -----------------------------
class InvestmentOutput(TypedDict, total=False):
    total_score: float  # 가중 평균 점수 (0-5.0)
    weighted_score: float  # total_score와 동일 (호환용)
    weights: Dict[str, float]
    scores: Dict[str, float]
    breakdown: Dict[str, Dict[str, Any]]  # 항목별 상세 정보
    decision: str
    rationale: str

class InvestmentState(TypedDict, total=False):
    tech_summary: Dict[str, Any]
    market_eval: Dict[str, Any]
    competition: Dict[str, Any]
    investment: InvestmentOutput

@dataclass
class InvestmentConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    weights: Dict[str, float] = None
    invest_threshold: float = 3.5  # 투자 기준 점수 (5점 만점)
    use_fintech_weights: bool = True  # fintech 가중치 사용 여부

    def __post_init__(self):
        if self.weights is None:
            if self.use_fintech_weights:
                # fintech 방식: 6개 항목, 0-5점 척도
                self.weights = {
                    "ROI": 0.2,
                    "기술_경쟁력": 0.2,
                    "시장성": 0.2,
                    "경쟁_우위": 0.15,
                    "팀_역량": 0.15,
                    "리스크": 0.1
                }
            else:
                # 기존 방식: 4개 항목, 0-100점 척도
                self.weights = {"tech": 0.30, "market": 0.25, "risk": 0.20, "edge": 0.25}


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

def _convert_score_to_5(score: float, max_val: float = 100) -> float:
    """0-100 점수를 0-5 척도로 변환"""
    return round(score / max_val * 5, 1)

def _llm_rationale(prompt: str, llm) -> str:
    if not llm:
        return (
            "기술·시장·리스크·경쟁 점수를 종합적으로 고려했을 때 "
            "기술 완성도와 차별화는 높고, 시장 성장성도 양호하나 "
            "규제와 보안 리스크는 지속적인 관리가 필요하다. "
            "따라서 중기적 관점에서 투자 검토가 유망하다."
        )
    try:
        chain = ChatPromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
        return chain.invoke({"prompt": prompt})
    except Exception:
        return (
            "기술·시장·리스크·경쟁 점수를 기반으로 한 자동 판단 결과: "
            "기술력과 경쟁우위는 긍정적이며 시장성도 충분하나, 리스크 관리 체계 강화가 필요하다."
        )

def _generate_summary_opinion(breakdown: Dict[str, Dict[str, Any]]) -> str:
    """요약 의견 생성 (fintech 방식)"""
    strengths = []
    
    if breakdown.get('ROI', {}).get('점수', 0) >= 4:
        strengths.append("ROI가 우수")
    if breakdown.get('기술_경쟁력', {}).get('점수', 0) >= 4:
        strengths.append("기술 경쟁력이 뛰어남")
    if breakdown.get('시장성', {}).get('점수', 0) >= 4:
        strengths.append("시장성이 높음")
    if breakdown.get('팀_역량', {}).get('점수', 0) >= 4:
        strengths.append("팀 역량이 우수")
    
    if strengths:
        opinion = f"{', '.join(strengths)}. "
    else:
        opinion = "전반적으로 양호한 수준. "
    
    return opinion


# ----------------------------- Node Factory -----------------------------
def _investment_node_factory(cfg: InvestmentConfig):
    def node(state: InvestmentState) -> InvestmentState:
        llm = _get_llm(cfg.model, cfg.temperature)
        
        if cfg.use_fintech_weights:
            # ===== fintech 방식: 가중치 기반 평가 (0-5점) =====
            
            # 점수 추출 (0-100 점수를 0-5로 변환)
            tech_raw = float(((state.get("tech_summary") or {}).get("tech_score", 0)) or 0)
            market_raw = float(((state.get("market_eval") or {}).get("market_score", 0)) or 0)
            risk_raw = float(((state.get("market_eval") or {}).get("risk_score", 0)) or 0)
            edge_raw = float(((state.get("competition") or {}).get("edge_score", 0)) or 0)
            
            # 0-5 척도로 변환
            roi_score = _convert_score_to_5(tech_raw)  # ROI는 기술 점수 기반 추정
            tech_score = _convert_score_to_5(tech_raw)
            market_score = _convert_score_to_5(market_raw)
            edge_score = _convert_score_to_5(edge_raw)
            team_score = 4.0  # 기본값 (상위 에이전트에서 제공 시 사용)
            risk_score = _convert_score_to_5(100 - risk_raw)  # 리스크는 역산
            
            # 가중 평균 계산
            weighted_score = (
                cfg.weights["ROI"] * roi_score +
                cfg.weights["기술_경쟁력"] * tech_score +
                cfg.weights["시장성"] * market_score +
                cfg.weights["경쟁_우위"] * edge_score +
                cfg.weights["팀_역량"] * team_score +
                cfg.weights["리스크"] * risk_score
            )
            weighted_score = round(weighted_score, 2)
            
            # 투자 판단
            if weighted_score >= cfg.invest_threshold:
                decision = "투자 적격 (Invest)"
            else:
                decision = "보류 (Hold)"
            
            # 항목별 상세 정보 (breakdown)
            criteria_description = {
                "ROI": "수익성",
                "기술_경쟁력": "특허, 기술력",
                "시장성": "시장 규모 및 성장률",
                "경쟁_우위": "차별화된 경쟁력",
                "팀_역량": "팀 구성 및 전문성",
                "리스크": "규제, 법률 리스크"
            }
            
            scores = {
                "ROI": roi_score,
                "기술_경쟁력": tech_score,
                "시장성": market_score,
                "경쟁_우위": edge_score,
                "팀_역량": team_score,
                "리스크": risk_score
            }
            
            breakdown = {}
            for criterion, score in scores.items():
                breakdown[criterion] = {
                    "점수": score,
                    "가중치": cfg.weights[criterion],
                    "가중_점수": round(score * cfg.weights[criterion], 2),
                    "설명": criteria_description[criterion]
                }
            
            # 요약 의견 생성
            opinion = _generate_summary_opinion(breakdown)
            if weighted_score >= cfg.invest_threshold:
                opinion += "투자 적격으로 판단됨."
            else:
                opinion += "보류 권장."
            
            # 결과 병합
            state["investment"] = {
                "total_score": weighted_score,
                "weighted_score": weighted_score,
                "weights": cfg.weights,
                "scores": scores,
                "breakdown": breakdown,
                "decision": decision,
                "rationale": opinion,
            }
            
            # 실행 결과 출력 (fintech 모드)
            company_name = (state.get("tech_summary") or {}).get("company", "N/A")
            print("\n" + "="*80)
            print("✅ [5/6] 투자 판단 완료 (Fintech 모드)")
            print("="*80)
            print(f"🏢 대상: {company_name}")
            print(f"📊 최종 점수: {weighted_score:.2f}/5.0")
            print(f"💡 결정: {decision}")
            print(f"📝 주요 항목:")
            for key, val in scores.items():
                print(f"   - {key}: {val:.1f}/5.0 (가중치 {cfg.weights.get(key, 0)*100:.0f}%)")
            print("="*80 + "\n")
            
        else:
            # ===== 기존 방식: 0-100점 척도 =====
            
            # 점수 추출
            tech = float(((state.get("tech_summary") or {}).get("tech_score", 0)) or 0)
            market = float(((state.get("market_eval") or {}).get("market_score", 0)) or 0)
            risk = float(((state.get("market_eval") or {}).get("risk_score", 0)) or 0)
            edge = float(((state.get("competition") or {}).get("edge_score", 0)) or 0)

            # 가중합 계산
            total = (
                cfg.weights["tech"] * tech
                + cfg.weights["market"] * market
                + cfg.weights["risk"] * risk
                + cfg.weights["edge"] * edge
            )
            total = round(total, 1)

            # 판단 레이블
            if total >= 80:
                decision = "INVEST"
            elif total >= 65:
                decision = "REVIEW"
            else:
                decision = "HOLD"

            # 프롬프트 구성
            prompt = f"""
너는 투자심사 애널리스트다.
아래 점수를 참고하여 {decision} 판단에 대한 근거를 5~7문장으로 명확하게 작성하라.

점수 요약:
기술력: {tech}, 시장성: {market}, 리스크 안정성: {risk}, 경쟁우위: {edge}, 총점: {total}
가중치: {json.dumps(cfg.weights)}

참고 요약:
기술 요약: {(state.get("tech_summary") or {}).get("summary", "")[:400]}
시장 요약: {(state.get("market_eval") or {}).get("market_summary", "")[:400]}
경쟁사 차별화: {(state.get("competition") or {}).get("differentiation_summary", "")[:400]}

형식:
"기술력은 ~, 시장성은 ~, 리스크는 ~, 경쟁우위는 ~. 따라서 {decision} 판단이 타당하다." 식의 간결한 설명.
"""
            rationale = _llm_rationale(prompt, llm).strip()

            # 결과 병합
            state["investment"] = {
                "total_score": total,
                "weights": cfg.weights,
                "scores": {"tech": tech, "market": market, "risk": risk, "edge": edge},
                "decision": decision,
                "rationale": rationale,
            }
            
            # 실행 결과 출력 (legacy 모드)
            company_name = (state.get("tech_summary") or {}).get("company", "N/A")
            print("\n" + "="*80)
            print("✅ [5/6] 투자 판단 완료 (Legacy 모드)")
            print("="*80)
            print(f"🏢 대상: {company_name}")
            print(f"📊 최종 점수: {total:.1f}/100")
            print(f"💡 결정: {decision}")
            print(f"📝 주요 항목:")
            print(f"   - 기술: {tech:.1f}/100 (가중치 {cfg.weights.get('tech', 0)*100:.0f}%)")
            print(f"   - 시장: {market:.1f}/100 (가중치 {cfg.weights.get('market', 0)*100:.0f}%)")
            print(f"   - 리스크: {risk:.1f}/100 (가중치 {cfg.weights.get('risk', 0)*100:.0f}%)")
            print(f"   - 경쟁우위: {edge:.1f}/100 (가중치 {cfg.weights.get('edge', 0)*100:.0f}%)")
            print("="*80 + "\n")

        return state

    return node


# ----------------------------- Graph Builder -----------------------------
def build_investment_graph(config: Optional[InvestmentConfig] = None):
    cfg = config or InvestmentConfig()
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph가 설치되어 있지 않습니다. `pip install langgraph` 후 다시 시도하세요.")
    g = StateGraph(InvestmentState)
    g.add_node("investment", _investment_node_factory(cfg))
    g.add_edge(START, "investment")
    g.add_edge("investment", END)
    return g.compile()


# ----------------------------- Helper for Direct Run -----------------------------
def run_investment_decision(state: Dict[str, Any], config: Optional[InvestmentConfig] = None) -> Dict[str, Any]:
    app = build_investment_graph(config)
    return app.invoke(state)


# ----------------------------- Output Formatting -----------------------------

def print_investment_summary(result: Dict[str, Any], use_fintech: bool = True):
    """투자 판단 결과를 보기 좋게 출력"""
    inv = result.get("investment", {})
    
    print("\n" + "=" * 80)
    print("🧮 투자 판단 결과")
    print("=" * 80)
    
    if use_fintech:
        print(f"\n가중 평균 점수: {inv.get('weighted_score', 0):.2f} / 5.0")
        print(f"투자 판단: {inv.get('decision', 'N/A')}")
        print(f"\n💡 판단 근거:\n{inv.get('rationale', '')}")
        
        if inv.get('breakdown'):
            print(f"\n{'='*80}")
            print("📊 항목별 평가 상세")
            print(f"{'='*80}")
            print(f"\n{'항목':<15} {'점수':>6} {'가중치':>8} {'가중점수':>10} {'설명':<20}")
            print("-" * 80)
            
            for criterion, details in inv['breakdown'].items():
                criterion_name = criterion.replace("_", " ")
                print(f"{criterion_name:<15} {details['점수']:>6.1f} {details['가중치']:>8.2f} "
                      f"{details['가중_점수']:>10.2f} {details['설명']:<20}")
            
            print(f"\n{'총합':<15} {'':<6} {'1.00':>8} {inv.get('weighted_score', 0):>10.2f}")
    else:
        print(f"\n총점: {inv.get('total_score', 0):.1f} / 100")
        print(f"투자 판단: {inv.get('decision', 'N/A')}")
        print(f"\n💡 판단 근거:\n{inv.get('rationale', '')}")
        
        if inv.get('scores'):
            print(f"\n{'='*80}")
            print("📊 항목별 점수")
            print(f"{'='*80}")
            for k, v in inv['scores'].items():
                weight = inv.get('weights', {}).get(k, 0)
                print(f"  {k}: {v:.1f} (가중치: {weight:.2f})")
    
    print("\n" + "=" * 80)


# ----------------------------- CLI Test -----------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("🧮 Investment Decision Agent 테스트")
    print("=" * 80)
    
    # 테스트 1: fintech 방식 (가중치 기반, 0-5점)
    print("\n[테스트 1] fintech 방식 - 가중치 기반 평가 (0-5점)")
    print("-" * 80)
    
    dummy_fintech: InvestmentState = {
        "tech_summary": {"tech_score": 82, "summary": "ML기반 신용평가 모델, 특허 2건 보유."},
        "market_eval": {"market_score": 79, "risk_score": 73, "market_summary": "사기탐지 시장 연 12% 성장."},
        "competition": {"edge_score": 76, "differentiation_summary": "모델 효율성 우위, 실시간 처리."},
    }
    
    config_fintech = InvestmentConfig(use_fintech_weights=True, invest_threshold=3.5)
    result_fintech = run_investment_decision(dummy_fintech, config_fintech)
    print_investment_summary(result_fintech, use_fintech=True)
    
    # 테스트 2: 기존 방식 (0-100점)
    print("\n" + "=" * 80)
    print("\n[테스트 2] 기존 방식 - 0-100점 척도")
    print("-" * 80)
    
    config_legacy = InvestmentConfig(use_fintech_weights=False)
    result_legacy = run_investment_decision(dummy_fintech, config_legacy)
    print_investment_summary(result_legacy, use_fintech=False)