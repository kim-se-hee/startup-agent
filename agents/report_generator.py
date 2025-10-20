"""
Module: agents/report_generator.py (LangGraph + PDF 통합 버전)
Purpose: 최종 투자 분석 보고서 생성 (Markdown + PDF + LangGraph 기반)

입력:
state = {
  "segment": "fintech ai ...",
  "startup_search": {"candidates": [...], "docs": [...]},
  "tech_summary": {"company": "Zest AI", "summary": "...", "strengths": [...], "weaknesses": [...], "tech_score": 82, "sources": [...]},
  "market_eval": {"market_summary": "...", "competitors": [...], "risk_summary": "...", "market_score": 79, "risk_score": 73, "sources": [...]},
  "competition": {"comparisons": [...], "differentiation_summary": "...", "edge_score": 76},
  "investment": {"total_score": 80.3, "weights": {...}, "scores": {...}, "decision": "INVEST", "rationale": "..."}
}

출력:
state["report"] = {
  "title": "<Company> 투자 분석 보고서",
  "markdown": "# ...",
  "filename": "investment_report_<Company>.md",
  "pdf_path": "outputs/investment_report_<Company>.pdf"  # PDF 추가
}
"""
from __future__ import annotations

import os
import datetime as dt
from typing import Any, Dict, List, Optional, TypedDict
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# PDF 생성을 위한 reportlab import
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False
    print("⚠️ reportlab이 설치되어 있지 않습니다. PDF 생성 기능이 비활성화됩니다.")

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


# ------------------------------ State & Config ------------------------------

class ReportOutput(TypedDict, total=False):
    title: str
    markdown: str
    filename: str
    pdf_path: str  # PDF 경로 추가

class ReportState(TypedDict, total=False):
    segment: str
    startup_search: Dict[str, Any]
    tech_summary: Dict[str, Any]
    market_eval: Dict[str, Any]
    competition: Dict[str, Any]
    investment: Dict[str, Any]
    report: ReportOutput

@dataclass
class ReportConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    use_llm_exec_summary: bool = False
    filename_prefix: str = "investment_report_"
    output_dir: str = "outputs"  # 출력 디렉토리
    generate_pdf: bool = True  # PDF 생성 여부


# ------------------------------ Helpers ------------------------------

def _norm(s: str) -> str:
    return (s or "").strip()

def _fmt_list(items: List[str]) -> str:
    return ", ".join([_norm(i) for i in items if _norm(i)]) if items else "-"

def _sources_union(state: Dict[str, Any], k: int = 10) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for section in ["startup_search", "market_eval", "competition"]:
        docs = []
        if section == "competition":
            for comp in (state.get("competition", {}) or {}).get("comparisons", []):
                docs += comp.get("sources", [])
        else:
            docs += (state.get(section, {}) or {}).get("docs", []) or (state.get(section, {}) or {}).get("sources", [])
        for d in docs[:k]:
            t, u = _norm(d.get("title", "")), _norm(d.get("url", ""))
            key = (t, u)
            if key not in seen and (t or u):
                out.append({"title": t, "url": u})
                seen.add(key)
            if len(out) >= k:
                return out
    return out[:k]

def _scores_table(scores: Dict[str, float], weights: Dict[str, float], total: float) -> str:
    """점수 테이블 생성 (fintech 모드와 레거시 모드 모두 지원)"""
    def _cell(v: float) -> str:
        try: return f"{float(v):.1f}"
        except Exception: return "-"
    
    # fintech 모드 확인 (ROI 키가 있으면 fintech)
    is_fintech_mode = "ROI" in scores or "기술_경쟁력" in scores
    
    lines = [
        "| 항목 | 점수 | 가중치 |",
        "|---|---:|---:|",
    ]
    
    if is_fintech_mode:
        # fintech 방식: 6개 항목 (0-5점)
        lines.extend([
            f"| ROI | {_cell(scores.get('ROI', 0))} | {int(weights.get('ROI', 0)*100)}% |",
            f"| 기술 경쟁력 | {_cell(scores.get('기술_경쟁력', 0))} | {int(weights.get('기술_경쟁력', 0)*100)}% |",
            f"| 시장성 | {_cell(scores.get('시장성', 0))} | {int(weights.get('시장성', 0)*100)}% |",
            f"| 경쟁 우위 | {_cell(scores.get('경쟁_우위', 0))} | {int(weights.get('경쟁_우위', 0)*100)}% |",
            f"| 팀 역량 | {_cell(scores.get('팀_역량', 0))} | {int(weights.get('팀_역량', 0)*100)}% |",
            f"| 리스크 | {_cell(scores.get('리스크', 0))} | {int(weights.get('리스크', 0)*100)}% |",
            f"| **총점** | **{_cell(total)}** /5.0 | — |",
        ])
    else:
        # 레거시 방식: 4개 항목 (0-100점)
        lines.extend([
            f"| 기술 경쟁력(tech) | {_cell(scores.get('tech', 0))} | {int(weights.get('tech', 0)*100)}% |",
            f"| 시장 성장성(market) | {_cell(scores.get('market', 0))} | {int(weights.get('market', 0)*100)}% |",
            f"| 리스크 안정성(risk) | {_cell(scores.get('risk', 0))} | {int(weights.get('risk', 0)*100)}% |",
            f"| 경쟁우위(edge) | {_cell(scores.get('edge', 0))} | {int(weights.get('edge', 0)*100)}% |",
            f"| **총점** | **{_cell(total)}** /100 | — |",
        ])
    
    return "\n".join(lines)


# ------------------------------ Optional LLM ------------------------------

def _get_llm(model: str, temperature: float):
    if not LANGCHAIN_CHAIN_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        return ChatOpenAI(model=model, temperature=temperature)
    except Exception:
        return None

def _refine_exec_summary(text: str, llm) -> str:
    if not llm:
        return text
    try:
        prompt = ChatPromptTemplate.from_template(
            """
다음 요약을 더 간결하고 임팩트 있게 3~5개의 불릿으로 한국어로 재작성하라.
불필요한 수식어는 제거하고 핵심 수치/강점/리스크/결론만 남겨라.

원문:
{src}
"""
        )
        chain = prompt | llm | StrOutputParser()
        refined = chain.invoke({"src": text})
        return refined.strip()
    except Exception:
        return text


# ------------------------------ PDF Generator (from fintech) ------------------------------

class PDFReportGenerator:
    """
    📋 PDF 투자 판단 보고서 생성기 (fintech/report_agent 기반)
    
    ai_agent_project의 분석 결과를 받아서 전문적인 PDF 보고서를 생성합니다.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        output_dir: PDF 파일이 저장될 디렉토리
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        if not REPORTLAB_AVAILABLE:
            print("⚠️ reportlab이 설치되지 않아 PDF를 생성할 수 없습니다.")
            self.korean_font = None
            return
        
        # 한글 폰트 등록 (AppleGothic for macOS)
        try:
            pdfmetrics.registerFont(TTFont('AppleGothic', '/System/Library/Fonts/Supplemental/AppleGothic.ttf'))
            self.korean_font = 'AppleGothic'
        except:
            print("⚠️ 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
            self.korean_font = 'Helvetica'
    
    def create_styles(self):
        """PDF 스타일 정의"""
        if not REPORTLAB_AVAILABLE:
            return None
            
        styles = getSampleStyleSheet()
        
        # 제목 스타일
        styles.add(ParagraphStyle(
            name='KoreanTitle',
            fontName=self.korean_font,
            fontSize=24,
            alignment=TA_CENTER,
            spaceAfter=30,
            textColor=colors.HexColor('#1a1a1a')
        ))
        
        # 서브타이틀 스타일
        styles.add(ParagraphStyle(
            name='KoreanSubtitle',
            fontName=self.korean_font,
            fontSize=14,
            alignment=TA_CENTER,
            spaceAfter=15,
            textColor=colors.HexColor('#34495e')
        ))
        
        # 부제목 스타일 (대분류)
        styles.add(ParagraphStyle(
            name='KoreanHeading',
            fontName=self.korean_font,
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor('#2c3e50'),
            bold=True
        ))
        
        # 소제목 스타일 (중분류)
        styles.add(ParagraphStyle(
            name='KoreanSubHeading',
            fontName=self.korean_font,
            fontSize=12,
            spaceBefore=10,
            spaceAfter=8,
            textColor=colors.HexColor('#34495e'),
            bold=True
        ))
        
        # 본문 텍스트 스타일
        styles.add(ParagraphStyle(
            name='KoreanBodyText',
            fontName=self.korean_font,
            fontSize=10,
            alignment=TA_LEFT,
            leading=16,
            spaceAfter=6
        ))
        
        # 표 안 텍스트 스타일 (줄바꿈 지원)
        styles.add(ParagraphStyle(
            name='KoreanTableText',
            fontName=self.korean_font,
            fontSize=10,
            alignment=TA_LEFT,
            leading=14
        ))
        
        return styles
    
    def _clean_html_text(self, text):
        """HTML 태그 제거 및 텍스트 정리"""
        if not text:
            return ""
        import re
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', str(text))
        # <br/> 태그를 실제 줄바꿈으로 변환
        text = text.replace('<br/>', '\n').replace('<br>', '\n')
        # 연속된 공백 정리
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _wrap_text_for_table(self, text, max_length=50):
        """표에서 텍스트 줄바꿈 처리"""
        if not text:
            return ""
        
        text = self._clean_html_text(text)
        if len(text) <= max_length:
            return text
        
        # 단어 단위로 줄바꿈
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= max_length:
                current_line += (" " + word) if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return '\n'.join(lines)
    
    def generate_pdf(self, state: Dict[str, Any], company_name: str, evaluator: str = "AI 투자분석팀") -> Optional[str]:
        """
        투자 분석 결과를 바탕으로 PDF 보고서 생성 (신규 목차 구조)
        
        Returns: 생성된 PDF 파일 경로 또는 None
        """
        if not REPORTLAB_AVAILABLE or not self.korean_font:
            print("⚠️ PDF 생성을 건너뜁니다.")
            return None
        
        try:
            # 파일명 생성
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"investment_report_{company_name.replace(' ', '_')}_{timestamp}.pdf"
            filepath = os.path.join(self.output_dir, filename)
            
            # PDF 문서 생성
            doc = SimpleDocTemplate(filepath, pagesize=A4,
                                   rightMargin=2*cm, leftMargin=2*cm,
                                   topMargin=2*cm, bottomMargin=2*cm)
            
            styles = self.create_styles()
            elements = []
            
            # 제목
            title = Paragraph("📑 AI 핀테크 스타트업 투자 분석 보고서", styles['KoreanTitle'])
            elements.append(title)
            elements.append(Spacer(1, 0.3*cm))
            
            subtitle = Paragraph(f"분석 대상: {company_name}", styles['KoreanSubtitle'])
            elements.append(subtitle)
            elements.append(Spacer(1, 0.5*cm))
            
            # 기본 정보
            date_str = dt.datetime.now().strftime("%Y년 %m월 %d일")
            info_data = [
                ['분석 대상', company_name],
                ['분석 일자', date_str],
                ['분석팀', evaluator],
                ['버전', 'v2.0']
            ]
            
            info_table = Table(info_data, colWidths=[5*cm, 12*cm])
            info_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), self.korean_font),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ]))
            
            elements.append(info_table)
            elements.append(Spacer(1, 0.8*cm))
            
            # 데이터 추출
            inv = state.get("investment", {}) or {}
            tech = state.get("tech_summary", {}) or {}
            market = state.get("market_eval", {}) or {}
            competition = state.get("competition", {}) or {}
            
            total_score = inv.get("total_score", 0)
            decision = inv.get("decision", "REVIEW")
            is_fintech = "ROI" in inv.get("scores", {}) or "기술_경쟁력" in inv.get("scores", {})
            
            # ═══════════════════════════════════════════════════════════
            # Ⅰ. 요약 (Executive Summary)
            # ═══════════════════════════════════════════════════════════
            elements.append(Paragraph("Ⅰ. 요약 (Executive Summary)", styles['KoreanHeading']))
            elements.append(Spacer(1, 0.3*cm))
            
            exec_summary_text = self._generate_executive_summary(state, company_name, decision, total_score, is_fintech)
            exec_para = Paragraph(exec_summary_text, styles['KoreanBodyText'])
            elements.append(exec_para)
            elements.append(Spacer(1, 0.8*cm))
            
            # ═══════════════════════════════════════════════════════════
            # Ⅱ. 기업 개요 (Company Overview)
            # ═══════════════════════════════════════════════════════════
            elements.append(Paragraph("Ⅱ. 기업 개요 (Company Overview)", styles['KoreanHeading']))
            elements.append(Spacer(1, 0.3*cm))
            
            company_overview = self._create_company_overview(state, company_name)
            for item in company_overview:
                elements.append(item)
            elements.append(Spacer(1, 0.8*cm))
            
            # ═══════════════════════════════════════════════════════════
            # Ⅲ. 기술 분석 (Technology Analysis)
            # ═══════════════════════════════════════════════════════════
            elements.append(Paragraph("Ⅲ. 기술 분석 (Technology Analysis)", styles['KoreanHeading']))
            elements.append(Spacer(1, 0.3*cm))
            
            tech_analysis = self._create_technology_analysis(state)
            for item in tech_analysis:
                elements.append(item)
            elements.append(Spacer(1, 0.8*cm))
            
            # ═══════════════════════════════════════════════════════════
            # Ⅳ. 시장 및 산업 환경
            # ═══════════════════════════════════════════════════════════
            elements.append(Paragraph("Ⅳ. 시장 및 산업 환경", styles['KoreanHeading']))
            elements.append(Spacer(1, 0.3*cm))
            
            market_analysis = self._create_market_environment(state)
            for item in market_analysis:
                elements.append(item)
            elements.append(Spacer(1, 0.8*cm))
            
            # ═══════════════════════════════════════════════════════════
            # Ⅴ. 경쟁사 비교
            # ═══════════════════════════════════════════════════════════
            elements.append(Paragraph("Ⅴ. 경쟁사 비교", styles['KoreanHeading']))
            elements.append(Spacer(1, 0.3*cm))
            
            competitor_analysis = self._create_competitor_comparison(state)
            for item in competitor_analysis:
                elements.append(item)
            elements.append(Spacer(1, 0.8*cm))
            
            # ═══════════════════════════════════════════════════════════
            # Ⅵ. 종합 투자 평가 (Investment Evaluation)
            # ═══════════════════════════════════════════════════════════
            elements.append(Paragraph("Ⅵ. 종합 투자 평가 (Investment Evaluation)", styles['KoreanHeading']))
            elements.append(Spacer(1, 0.3*cm))
            
            investment_eval = self._create_investment_evaluation_table(state, is_fintech)
            elements.append(investment_eval)
            elements.append(Spacer(1, 0.8*cm))
            
            # ═══════════════════════════════════════════════════════════
            # Ⅶ. 투자 판단 및 제언 (Conclusion & Recommendation)
            # ═══════════════════════════════════════════════════════════
            elements.append(Paragraph("Ⅶ. 투자 판단 및 제언", styles['KoreanHeading']))
            elements.append(Spacer(1, 0.3*cm))
            
            conclusion = self._create_conclusion_section(state, decision, is_fintech)
            for item in conclusion:
                elements.append(item)
            elements.append(Spacer(1, 0.8*cm))
            
            # ═══════════════════════════════════════════════════════════
            # Ⅷ. 부록 (Appendix)
            # ═══════════════════════════════════════════════════════════
            elements.append(Paragraph("Ⅷ. 부록 (Appendix)", styles['KoreanHeading']))
            elements.append(Spacer(1, 0.3*cm))
            
            appendix = self._create_appendix(state)
            for item in appendix:
                elements.append(item)
            
            # PDF 빌드
            doc.build(elements)
            
            print(f"✅ PDF 보고서가 생성되었습니다: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ PDF 생성 중 오류 발생: {e}")
            return None
    
    def _get_table_style(self):
        """공통 테이블 스타일"""
        return TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), self.korean_font),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
            ('WORDWRAP', (0, 0), (-1, -1), 'CJK')
        ])
    
    def _generate_executive_summary(self, state, company_name, decision, total_score, is_fintech):
        """Ⅰ. 요약 생성"""
        tech = state.get("tech_summary", {}) or {}
        market = state.get("market_eval", {}) or {}
        inv = state.get("investment", {}) or {}
        
        score_text = f"{total_score:.2f}/5.0" if is_fintech else f"{total_score:.1f}/100"
        
        # 핵심 투자 포인트
        key_points = []
        if tech.get("strengths"):
            key_points.append(f"기술 강점: {', '.join(tech['strengths'][:2])}")
        if market.get("key_insights"):
            key_points.append(f"시장 인사이트: {market['key_insights'][0][:50]}...")
        
        # 주요 리스크
        risks = []
        if tech.get("weaknesses"):
            risks.append(tech["weaknesses"][0])
        if market.get("risk_factors"):
            risks.append(market["risk_factors"][0])
        
        key_points_text = '\n'.join([f"• {self._clean_html_text(p)}" for p in key_points]) if key_points else "• 양호한 기술력 및 시장 잠재력"
        risks_text = '\n'.join([f"• {self._clean_html_text(r)[:60]}" for r in risks[:2]]) if risks else "• 일반적인 핀테크 규제 리스크"
        rationale_text = self._clean_html_text(inv.get('rationale', '종합적으로 양호한 투자 기회로 평가됩니다.'))
        
        # 종합 의견에서 쉼표로 구분된 항목들을 줄바꿈으로 분리
        if rationale_text and ',' in rationale_text:
            rationale_items = [item.strip() for item in rationale_text.split(',') if item.strip()]
            rationale_text = '\n'.join([f"• {item}" for item in rationale_items])
        else:
            rationale_text = f"• {rationale_text}"
        
        summary = f"""<b>【분석 대상】</b><br/>
{company_name}<br/><br/>

<b>【최종 판단】</b><br/>
{decision} (점수: {score_text})<br/><br/>

<b>【핵심 투자 포인트】</b><br/>
{key_points_text}<br/><br/>

<b>【주요 리스크】</b><br/>
{risks_text}<br/><br/>

<b>【종합 의견】</b><br/>
{rationale_text}
"""
        return summary
    
    def _create_company_overview(self, state, company_name):
        """Ⅱ. 기업 개요"""
        tech = state.get("tech_summary", {}) or {}
        startup_search = state.get("startup_search", {}) or {}
        
        # 회사 정보 찾기
        company_desc = ""
        company_domain = ""
        
        candidates = startup_search.get("candidates", [])
        for c in candidates:
            if c.get("name", "").lower() == company_name.lower():
                company_desc = c.get("description", "")
                company_domain = c.get("domain", "")
                break
        
        if not company_desc:
            company_desc = tech.get("summary", "핀테크 AI 기술을 활용하는 스타트업")
        
        company_detail = tech.get("company_detail", {})
        if company_detail:
            company_info = company_detail.get("company", {})
            company_domain = company_info.get("domain", company_domain)
            if not company_desc:
                company_desc = company_info.get("desription", "")
        
        styles = self.create_styles()
        items = []
        
        # 텍스트 정리 및 줄바꿈 처리
        company_desc_clean = self._clean_html_text(company_desc)
        company_desc_clean = company_desc_clean[:200] + "..." if len(company_desc_clean) > 200 else company_desc_clean
        
        # 기업 개요를 Paragraph로 감싸서 줄바꿈 처리
        styles = self.create_styles()
        company_desc_para = Paragraph(company_desc_clean, styles['KoreanTableText'])
        
        data = [
            ['회사명', company_name],
            ['세부 도메인', company_domain or "Fintech/AI"],
            ['기업 개요', company_desc_para]
        ]
        
        table = Table(data, colWidths=[5*cm, 12*cm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), self.korean_font),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        
        items.append(table)
        return items
    
    def _create_technology_analysis(self, state):
        """Ⅲ. 기술 분석"""
        tech = state.get("tech_summary", {}) or {}
        styles = self.create_styles()
        items = []
        
        # 자사 서비스 목록
        company_detail = tech.get("company_detail", {})
        products = company_detail.get("products", [])
        
        if products:
            items.append(Paragraph("<b>• 자사 서비스 목록</b>", styles['KoreanBodyText']))
            items.append(Spacer(1, 0.2*cm))
            
            for idx, prod in enumerate(products, 1):
                prod_name = prod.get("name", f"제품 {idx}")
                prod_desc = self._clean_html_text(prod.get("description", ""))
                prod_text = f"{idx}. <b>{prod_name}</b>: {prod_desc[:100]}"
                items.append(Paragraph(prod_text, styles['KoreanBodyText']))
                items.append(Spacer(1, 0.1*cm))
            
            items.append(Spacer(1, 0.3*cm))
            
            # 활용된 AI 원천 기술
            items.append(Paragraph("<b>• 활용된 AI 원천 기술</b>", styles['KoreanBodyText']))
            items.append(Spacer(1, 0.2*cm))
            
            for prod in products[:2]:
                tech_desc = self._clean_html_text(prod.get("description", ""))
                if tech_desc:
                    items.append(Paragraph(f"- {tech_desc[:150]}", styles['KoreanBodyText']))
                    items.append(Spacer(1, 0.1*cm))
        
        # 강점 및 한계
        strengths = tech.get("strengths", [])
        weaknesses = tech.get("weaknesses", [])
        
        if strengths or weaknesses:
            items.append(Spacer(1, 0.3*cm))
            
            tech_eval_data = []
            if strengths:
                strengths_clean = [self._clean_html_text(s) for s in strengths[:3]]
                # 쉼표로 구분된 항목들을 줄바꿈으로 분리
                strength_text = []
                for s in strengths_clean:
                    if ',' in s:
                        items = [item.strip() for item in s.split(',') if item.strip()]
                        strength_text.extend([f"• {item}" for item in items])
                    else:
                        strength_text.append(f"• {s}")
                tech_eval_data.append(['타 서비스 대비 강점', '\n'.join(strength_text)])
            if weaknesses:
                weaknesses_clean = [self._clean_html_text(w) for w in weaknesses[:3]]
                # 쉼표로 구분된 항목들을 줄바꿈으로 분리
                weakness_text = []
                for w in weaknesses_clean:
                    if ',' in w:
                        items = [item.strip() for item in w.split(',') if item.strip()]
                        weakness_text.extend([f"• {item}" for item in items])
                    else:
                        weakness_text.append(f"• {w}")
                tech_eval_data.append(['한계 및 리스크 요인', '\n'.join(weakness_text)])
            
            if tech_eval_data:
                # 각 항목을 Paragraph로 감싸서 줄바꿈 처리
                tech_eval_data_para = []
                for row in tech_eval_data:
                    para_row = [row[0], Paragraph(row[1], styles['KoreanTableText'])]
                    tech_eval_data_para.append(para_row)
                
                tech_eval_table = Table(tech_eval_data_para, colWidths=[5*cm, 12*cm])
                tech_eval_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (-1, -1), self.korean_font),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#d5dbdb')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ]))
                items.append(tech_eval_table)
        
        return items
    
    def _create_market_environment(self, state):
        """Ⅳ. 시장 및 산업 환경"""
        market = state.get("market_eval", {}) or {}
        styles = self.create_styles()
        items = []
        
        # 시장 규모와 성장 전망
        items.append(Paragraph("<b>시장 규모와 성장 전망</b>", styles['KoreanSubHeading']))
        items.append(Spacer(1, 0.2*cm))
        
        market_size = market.get("market_size", {})
        growth = market.get("growth_potential", {})
        
        size_data = []
        if market_size and isinstance(market_size, dict):
            tam = market_size.get("TAM")
            sam = market_size.get("SAM")
            som = market_size.get("SOM")
            if tam:
                size_data.append(['TAM (전체 시장)', f"{tam:,.0f} {market_size.get('currency', 'KRW')}"])
            if sam:
                size_data.append(['SAM (접근 가능 시장)', f"{sam:,.0f} {market_size.get('currency', 'KRW')}"])
            if som:
                size_data.append(['SOM (점유 가능 시장)', f"{som:,.0f} {market_size.get('currency', 'KRW')}"])
        
        if growth and isinstance(growth, dict):
            cagr = growth.get("CAGR")
            if cagr:
                size_data.append(['CAGR (연평균 성장률)', f"{cagr:.1f}%"])
            drivers = growth.get("growth_drivers", [])
            if drivers:
                size_data.append(['핵심 성장 동인', ', '.join(drivers[:3])])
        
        if not size_data:
            size_data = [['시장 요약', market.get("market_summary", "핀테크 시장 성장 중")]]
        
        market_size_table = Table(size_data, colWidths=[6*cm, 11*cm])
        market_size_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), self.korean_font),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#d5dbdb')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        items.append(market_size_table)
        items.append(Spacer(1, 0.5*cm))
        
        # 경쟁 구도 및 차별화
        comp_landscape = market.get("competitive_landscape", {})
        if comp_landscape and isinstance(comp_landscape, dict):
            items.append(Paragraph("<b>경쟁 구도 및 차별화 요인</b>", styles['KoreanSubHeading']))
            items.append(Spacer(1, 0.2*cm))
            
            comp_data = []
            intensity = comp_landscape.get("intensity")
            if intensity:
                comp_data.append(['경쟁 강도', intensity])
            
            key_players = comp_landscape.get("key_players", [])
            if key_players:
                comp_data.append(['주요 경쟁사', ', '.join(key_players[:4])])
            
            diff = comp_landscape.get("differentiation")
            if diff:
                comp_data.append(['차별화 포인트', diff[:100]])
            
            if comp_data:
                comp_table = Table(comp_data, colWidths=[6*cm, 11*cm])
                comp_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (-1, -1), self.korean_font),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#d5dbdb')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ]))
                items.append(comp_table)
                items.append(Spacer(1, 0.5*cm))
        
        # 규제 및 정책 환경
        regulatory = market.get("regulatory_environment", {})
        if regulatory and isinstance(regulatory, dict):
            items.append(Paragraph("<b>규제 및 정책 환경</b>", styles['KoreanSubHeading']))
            items.append(Spacer(1, 0.2*cm))
            
            reg_data = []
            risk_level = regulatory.get("risk_level")
            if risk_level:
                reg_data.append(['규제 리스크 수준', risk_level])
            
            key_regs = regulatory.get("key_regulations", [])
            if key_regs:
                reg_data.append(['주요 규제', ', '.join(key_regs[:3])])
            
            if reg_data:
                reg_table = Table(reg_data, colWidths=[6*cm, 11*cm])
                reg_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (-1, -1), self.korean_font),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#d5dbdb')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ]))
                items.append(reg_table)
        
        # 시장 인사이트 요약
        if market.get("key_insights"):
            items.append(Spacer(1, 0.5*cm))
            items.append(Paragraph("<b>시장 인사이트 요약</b>", styles['KoreanSubHeading']))
            items.append(Spacer(1, 0.2*cm))
            
            insights_text = '<br/>'.join([f"• {insight[:80]}" for insight in market["key_insights"][:3]])
            items.append(Paragraph(insights_text, styles['KoreanBodyText']))
        
        return items
    
    def _create_competitor_comparison(self, state):
        """Ⅴ. 경쟁사 비교"""
        competition = state.get("competition", {}) or {}
        styles = self.create_styles()
        items = []
        
        comparisons = competition.get("comparisons", [])
        edge_score = competition.get("edge_score", 0)
        
        # 경쟁우위 점수
        score_para = Paragraph(f"<b>경쟁우위 점수:</b> {edge_score:.1f}/100", styles['KoreanBodyText'])
        items.append(score_para)
        items.append(Spacer(1, 0.3*cm))
        
        if comparisons:
            comp_data = [['경쟁사', '강점', '약점', '포지셔닝']]
            
            for comp in comparisons[:5]:
                name = comp.get("name", "N/A")
                strengths_clean = [self._clean_html_text(s) for s in comp.get("strengths", [])[:2]]
                weaknesses_clean = [self._clean_html_text(w) for w in comp.get("weaknesses", [])[:2]]
                positioning_clean = self._clean_html_text(comp.get("positioning", ""))[:30]
                
                # 강점과 약점에서 쉼표로 구분된 항목들을 줄바꿈으로 분리
                strength_items = []
                for s in strengths_clean:
                    if ',' in s:
                        items = [item.strip() for item in s.split(',') if item.strip()]
                        strength_items.extend(items)
                    else:
                        strength_items.append(s)
                
                weakness_items = []
                for w in weaknesses_clean:
                    if ',' in w:
                        items = [item.strip() for item in w.split(',') if item.strip()]
                        weakness_items.extend(items)
                    else:
                        weakness_items.append(w)
                
                comp_data.append([
                    name, 
                    '\n'.join([f"• {item}" for item in strength_items[:3]]), 
                    '\n'.join([f"• {item}" for item in weakness_items[:3]]), 
                    positioning_clean
                ])
            
            # 각 셀을 Paragraph로 감싸서 줄바꿈 처리
            comp_data_para = []
            for i, row in enumerate(comp_data):
                if i == 0:  # 헤더 행
                    comp_data_para.append(row)
                else:
                    para_row = [
                        row[0],  # 경쟁사명
                        Paragraph(row[1], styles['KoreanTableText']),  # 강점
                        Paragraph(row[2], styles['KoreanTableText']),  # 약점
                        Paragraph(row[3], styles['KoreanTableText'])   # 포지셔닝
                    ]
                    comp_data_para.append(para_row)
            
            comp_table = Table(comp_data_para, colWidths=[4*cm, 4*cm, 4*cm, 5*cm])
            comp_table.setStyle(self._get_table_style())
            items.append(comp_table)
            items.append(Spacer(1, 0.3*cm))
        
        # 차별화 전략 요약
        diff_summary = self._clean_html_text(competition.get("differentiation_summary", ""))
        if diff_summary:
            diff_para = Paragraph(f"<b>차별화 전략:</b> {diff_summary[:150]}", styles['KoreanBodyText'])
            items.append(diff_para)
        
        return items
    
    def _create_investment_evaluation_table(self, state, is_fintech):
        """Ⅵ. 종합 투자 평가 (테이블)"""
        inv = state.get("investment", {}) or {}
        scores = inv.get("scores", {})
        weights = inv.get("weights", {})
        total_score = inv.get("total_score", 0)
        
        if is_fintech:
            # fintech 방식: 6개 항목
            eval_data = [
                ['항목', '세부 평가 포인트', '점수', '가중치', '가중 합산'],
                ['기술 경쟁력', '기술 성숙도', f"{scores.get('기술_경쟁력', 0):.1f}/5.0", f"{weights.get('기술_경쟁력', 0)*100:.0f}%", f"{scores.get('기술_경쟁력', 0) * weights.get('기술_경쟁력', 0):.2f}"],
                ['ROI', '수익성', f"{scores.get('ROI', 0):.1f}/5.0", f"{weights.get('ROI', 0)*100:.0f}%", f"{scores.get('ROI', 0) * weights.get('ROI', 0):.2f}"],
                ['시장성', '성장률, 진입 기회', f"{scores.get('시장성', 0):.1f}/5.0", f"{weights.get('시장성', 0)*100:.0f}%", f"{scores.get('시장성', 0) * weights.get('시장성', 0):.2f}"],
                ['경쟁 우위', '차별화, 진입장벽', f"{scores.get('경쟁_우위', 0):.1f}/5.0", f"{weights.get('경쟁_우위', 0)*100:.0f}%", f"{scores.get('경쟁_우위', 0) * weights.get('경쟁_우위', 0):.2f}"],
                ['팀 역량', '팀 구성 및 전문성', f"{scores.get('팀_역량', 0):.1f}/5.0", f"{weights.get('팀_역량', 0)*100:.0f}%", f"{scores.get('팀_역량', 0) * weights.get('팀_역량', 0):.2f}"],
                ['규제 리스크', '보안·법률 리스크', f"{scores.get('리스크', 0):.1f}/5.0", f"{weights.get('리스크', 0)*100:.0f}%", f"{scores.get('리스크', 0) * weights.get('리스크', 0):.2f}"],
                ['', '종합 투자 점수', '', '', f"{total_score:.2f}/5.0"]
            ]
        else:
            # legacy 방식: 4개 항목
            eval_data = [
                ['항목', '세부 평가 포인트', '점수', '가중치', '가중 합산'],
                ['기술 경쟁력', '기술 성숙도', f"{scores.get('tech', 0):.0f}/100", f"{weights.get('tech', 0)*100:.0f}%", f"{scores.get('tech', 0) * weights.get('tech', 0):.1f}"],
                ['시장성', '성장률, 진입 기회', f"{scores.get('market', 0):.0f}/100", f"{weights.get('market', 0)*100:.0f}%", f"{scores.get('market', 0) * weights.get('market', 0):.1f}"],
                ['경쟁력', '차별화, 진입장벽', f"{scores.get('edge', 0):.0f}/100", f"{weights.get('edge', 0)*100:.0f}%", f"{scores.get('edge', 0) * weights.get('edge', 0):.1f}"],
                ['리스크', '규제·법률 리스크', f"{scores.get('risk', 0):.0f}/100", f"{weights.get('risk', 0)*100:.0f}%", f"{scores.get('risk', 0) * weights.get('risk', 0):.1f}"],
                ['', '종합 투자 점수', '', '', f"{total_score:.1f}/100"]
            ]
        
        eval_table = Table(eval_data, colWidths=[3.5*cm, 4*cm, 3*cm, 2.5*cm, 4*cm])
        eval_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), self.korean_font),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, -1), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor('#f9f9f9')]),
            ('FONTSIZE', (0, -1), (-1, -1), 11),
            ('SPAN', (0, -1), (3, -1)),
        ]))
        
        return eval_table
    
    def _create_conclusion_section(self, state, decision, is_fintech):
        """Ⅶ. 투자 판단 및 제언"""
        inv = state.get("investment", {}) or {}
        market = state.get("market_eval", {}) or {}
        styles = self.create_styles()
        items = []
        
        # 투자 판단 근거
        rationale = self._clean_html_text(inv.get("rationale", "종합적으로 양호한 투자 기회로 판단됩니다."))
        items.append(Paragraph(f"<b>투자 판단 근거:</b><br/>{rationale}", styles['KoreanBodyText']))
        items.append(Spacer(1, 0.3*cm))
        
        # 잠재 성장 요인 vs 리스크 요인
        opportunities = market.get("opportunities", [])
        risks = market.get("risk_factors", [])
        
        if opportunities or risks:
            vs_data = []
            if opportunities:
                opp_clean = [self._clean_html_text(o) for o in opportunities[:3]]
                # 쉼표로 구분된 항목들을 줄바꿈으로 분리
                opp_text = []
                for o in opp_clean:
                    if ',' in o:
                        items = [item.strip() for item in o.split(',') if item.strip()]
                        opp_text.extend([f"• {item[:60]}" for item in items])
                    else:
                        opp_text.append(f"• {o[:60]}")
                vs_data.append(['잠재 성장 요인', '\n'.join(opp_text)])
            if risks:
                risk_clean = [self._clean_html_text(r) for r in risks[:3]]
                # 쉼표로 구분된 항목들을 줄바꿈으로 분리
                risk_text = []
                for r in risk_clean:
                    if ',' in r:
                        items = [item.strip() for item in r.split(',') if item.strip()]
                        risk_text.extend([f"• {item[:60]}" for item in items])
                    else:
                        risk_text.append(f"• {r[:60]}")
                vs_data.append(['리스크 요인', '\n'.join(risk_text)])
            
            if vs_data:
                # 각 항목을 Paragraph로 감싸서 줄바꿈 처리
                vs_data_para = []
                for row in vs_data:
                    para_row = [row[0], Paragraph(row[1], styles['KoreanTableText'])]
                    vs_data_para.append(para_row)
                
                vs_table = Table(vs_data_para, colWidths=[5*cm, 12*cm])
                vs_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (-1, -1), self.korean_font),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#d5dbdb')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ]))
                items.append(vs_table)
                items.append(Spacer(1, 0.3*cm))
        
        # 권장 투자 전략
        total_score = inv.get("total_score", 0)
        strategy = ""
        
        if is_fintech:
            if total_score >= 4.0:
                strategy = "강력추천: 적극적 투자 검토 (Series A/B 참여 고려)"
            elif total_score >= 3.5:
                strategy = "추천: 소규모 시드 투자 또는 전략적 파트너십 고려"
            else:
                strategy = "보류: 추가 모니터링 후 재평가"
        else:
            if total_score >= 70:
                strategy = "추천: 투자 검토 권장"
            else:
                strategy = "보류: 리스크 완화 후 재검토"
        
        items.append(Paragraph(f"<b>권장 투자 전략:</b> {strategy}", styles['KoreanBodyText']))
        items.append(Spacer(1, 0.3*cm))
        
        # 향후 모니터링 포인트
        monitoring = [
            "기술 개발 진척도 및 특허 출원 현황",
            "주요 금융기관 파트너십 확대",
            "규제 변화 및 컴플라이언스 대응"
        ]
        monitoring_text = '<br/>'.join([f"• {m}" for m in monitoring])
        items.append(Paragraph(f"<b>향후 모니터링 포인트:</b><br/>{monitoring_text}", styles['KoreanBodyText']))
        
        return items
    
    def _create_appendix(self, state):
        """Ⅷ. 부록"""
        styles = self.create_styles()
        items = []
        
        # 데이터 출처
        all_sources = []
        
        tech_sources = (state.get("tech_summary", {}) or {}).get("sources", [])
        market = state.get("market_eval", {}) or {}
        market_sources = market.get("sources", [])
        
        for src in tech_sources + market_sources:
            if isinstance(src, dict):
                title = self._clean_html_text(src.get("title", ""))
                url = src.get("url", "")
                if title:
                    all_sources.append(f"{title} - {url}" if url else title)
        
        if market.get("data_sources"):
            all_sources.extend(market["data_sources"][:5])
        
        if all_sources:
            items.append(Paragraph("<b>데이터 출처 목록:</b>", styles['KoreanSubHeading']))
            items.append(Spacer(1, 0.2*cm))
            
            sources_clean = [self._clean_html_text(s) for s in list(set(all_sources))[:10]]
            sources_text = '\n'.join([f"• {s[:100]}" for s in sources_clean])
            items.append(Paragraph(sources_text, styles['KoreanBodyText']))
            items.append(Spacer(1, 0.3*cm))
        
        # 분석 시점 및 한계
        date_str = dt.datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
        
        # 분석 시점
        analysis_time = f"<b>분석 시점:</b> {date_str}"
        items.append(Paragraph(analysis_time, styles['KoreanBodyText']))
        items.append(Spacer(1, 0.3*cm))
        
        # 분석 한계 - 각 항목을 별도로 표시
        items.append(Paragraph("<b>분석 한계:</b>", styles['KoreanBodyText']))
        items.append(Spacer(1, 0.2*cm))
        
        limitations_items = [
            "• 본 보고서는 AI 에이전트 기반 자동 분석으로 생성되었습니다.",
            "• 공개 데이터 및 웹 검색 결과를 기반으로 하므로, 비공개 정보는 반영되지 않습니다.",
            "• 투자 판단은 최종 의사결정이 아닌 참고 자료로 활용하시기 바랍니다.",
            "• 시장 및 규제 환경은 빠르게 변화할 수 있으므로 지속적인 모니터링이 필요합니다."
        ]
        
        for item in limitations_items:
            items.append(Paragraph(item, styles['KoreanBodyText']))
            items.append(Spacer(1, 0.1*cm))
        
        return items
    
    def _generate_summary_opinion(self, state: Dict[str, Any]) -> str:
        """요약 의견 생성 (fintech/레거시 모드 자동 감지)"""
        tech = state.get("tech_summary", {}) or {}
        market = state.get("market_eval", {}) or {}
        inv = state.get("investment", {}) or {}
        
        # fintech 모드 확인
        breakdown = inv.get("breakdown", {})
        
        if breakdown:
            # fintech 방식: breakdown 사용
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
            
            if "투자 적격" in inv.get("decision", ""):
                opinion += "투자 적격으로 판단됨."
            else:
                opinion += "보류 권장."
        else:
            # 레거시 방식: 기술/시장 점수 사용
            tech_score = tech.get("tech_score", 0)
            market_score = market.get("market_score", 0)
            decision = inv.get("decision", "REVIEW")
            
            strengths = []
            if tech_score >= 75:
                strengths.append("기술 경쟁력이 우수")
            if market_score >= 75:
                strengths.append("시장성이 높음")
            
            if strengths:
                opinion = f"{', '.join(strengths)}. "
            else:
                opinion = "전반적으로 양호한 수준. "
            
            if "INVEST" in decision:
                opinion += "투자 적격으로 판단됨."
            else:
                opinion += "추가 검토 필요."
        
        return opinion
    
    def _create_tech_table(self, tech: Dict, inv: Dict):
        """기술 분석 테이블 (fintech/레거시 자동 감지)"""
        scores = inv.get("scores", {})
        is_fintech = "기술_경쟁력" in scores
        
        if is_fintech:
            # fintech 방식
            tech_score = scores.get("기술_경쟁력", 0)
            weight = inv.get("weights", {}).get("기술_경쟁력", 0.2)
            data = [
                ['항목', '가중치', '점수(0~5)', '설명'],
                ['기술 경쟁력', f"{weight:.2f}", f"{tech_score:.1f}", _fmt_list(tech.get("strengths", []))],
            ]
            weighted = tech_score * weight
            data.append(['기술 평가 합계', f"{weight:.2f}", f"{weighted:.2f}", '우수' if tech_score >= 4 else '보통'])
        else:
            # 레거시 방식
            tech_score = tech.get("tech_score", 0)
            weight = inv.get("weights", {}).get("tech", 0.3)
            data = [
                ['항목', '가중치', '점수(0~100)', '설명'],
                ['기술 경쟁력', f"{weight:.2f}", f"{tech_score:.1f}", _fmt_list(tech.get("strengths", []))],
            ]
            weighted = tech_score * weight
            data.append(['기술 평가 합계', f"{weight:.2f}", f"{weighted:.1f}", '우수' if tech_score >= 75 else '보통'])
        
        return data
    
    def _create_market_table(self, market: Dict, inv: Dict):
        """시장 분석 테이블 (fintech/레거시 자동 감지)"""
        scores = inv.get("scores", {})
        is_fintech = "시장성" in scores
        
        if is_fintech:
            # fintech 방식
            market_score = scores.get("시장성", 0)
            weight = inv.get("weights", {}).get("시장성", 0.2)
            data = [
                ['항목', '가중치', '점수(0~5)', '설명'],
                ['시장성', f"{weight:.2f}", f"{market_score:.1f}", '시장 규모 및 성장률'],
            ]
            weighted = market_score * weight
            data.append(['시장 평가 합계', f"{weight:.2f}", f"{weighted:.2f}", '우수' if market_score >= 4 else '보통'])
        else:
            # 레거시 방식
            market_score = market.get("market_score", 0)
            weight = inv.get("weights", {}).get("market", 0.25)
            data = [
                ['항목', '가중치', '점수(0~100)', '설명'],
                ['시장 성장성', f"{weight:.2f}", f"{market_score:.1f}", '시장 규모 및 성장률'],
            ]
            weighted = market_score * weight
            data.append(['시장 평가 합계', f"{weight:.2f}", f"{weighted:.1f}", '우수' if market_score >= 75 else '보통'])
        
        return data
    
    def _create_risk_table(self, market: Dict, inv: Dict):
        """리스크 평가 테이블 (fintech/레거시 자동 감지)"""
        scores = inv.get("scores", {})
        is_fintech = "리스크" in scores
        
        if is_fintech:
            # fintech 방식
            risk_score = scores.get("리스크", 0)
            weight = inv.get("weights", {}).get("리스크", 0.1)
            data = [
                ['항목', '영향도', '점수(0~5)', '설명'],
                ['리스크 관리', '중', f"{risk_score:.1f}", '규제 및 운영 리스크'],
            ]
            weighted = risk_score * weight
            risk_level = "낮음" if risk_score >= 4 else ("보통" if risk_score >= 3 else "높음")
            data.append(['리스크 평균', '-', f"{weighted:.2f}", f"위험도 {risk_level}"])
        else:
            # 레거시 방식
            risk_score = market.get("risk_score", 0)
            weight = inv.get("weights", {}).get("risk", 0.2)
            data = [
                ['항목', '영향도', '점수(0~100)', '설명'],
                ['리스크 안정성', '중', f"{risk_score:.1f}", '규제 및 운영 리스크'],
            ]
            weighted = risk_score * weight
            risk_level = "낮음" if risk_score >= 75 else ("보통" if risk_score >= 60 else "높음")
            data.append(['리스크 평균', '-', f"{weighted:.1f}", f"위험도 {risk_level}"])
        
        return data
    
    def _create_final_table(self, inv: Dict):
        """종합 평가 테이블 (fintech/레거시 모드 자동 감지)"""
        scores = inv.get("scores", {})
        weights = inv.get("weights", {})
        total = inv.get("total_score", 0)
        decision = inv.get("decision", "REVIEW")
        
        # fintech 모드 확인
        is_fintech_mode = "ROI" in scores or "기술_경쟁력" in scores
        
        data = [['구분', '점수', '평가']]
        
        if is_fintech_mode:
            # fintech 방식: 6개 항목 (0-5점)
            data.extend([
                ['ROI', f"{scores.get('ROI', 0):.1f} / 5", self._get_rating_fintech(scores.get('ROI', 0))],
                ['기술 경쟁력', f"{scores.get('기술_경쟁력', 0):.1f} / 5", self._get_rating_fintech(scores.get('기술_경쟁력', 0))],
                ['시장성', f"{scores.get('시장성', 0):.1f} / 5", self._get_rating_fintech(scores.get('시장성', 0))],
                ['경쟁 우위', f"{scores.get('경쟁_우위', 0):.1f} / 5", self._get_rating_fintech(scores.get('경쟁_우위', 0))],
                ['팀 역량', f"{scores.get('팀_역량', 0):.1f} / 5", self._get_rating_fintech(scores.get('팀_역량', 0))],
                ['리스크', f"{scores.get('리스크', 0):.1f} / 5", self._get_rating_fintech(scores.get('리스크', 0))],
                ['총점', f"{total:.2f} / 5.0", decision]
            ])
        else:
            # 레거시 방식: 4개 항목 (0-100점)
            data.extend([
                ['기술 경쟁력', f"{scores.get('tech', 0):.1f} / 100", self._get_rating(scores.get('tech', 0))],
                ['시장 성장성', f"{scores.get('market', 0):.1f} / 100", self._get_rating(scores.get('market', 0))],
                ['리스크 안정성', f"{scores.get('risk', 0):.1f} / 100", self._get_rating(scores.get('risk', 0))],
                ['경쟁 우위', f"{scores.get('edge', 0):.1f} / 100", self._get_rating(scores.get('edge', 0))],
                ['총점', f"{total:.1f} / 100", decision]
            ])
        
        return data
    
    def _get_rating(self, score: float) -> str:
        """점수에 따른 등급 (0-100점)"""
        if score >= 80:
            return "우수"
        elif score >= 70:
            return "양호"
        else:
            return "보통"
    
    def _get_rating_fintech(self, score: float) -> str:
        """점수에 따른 등급 (0-5점)"""
        if score >= 4.0:
            return "우수"
        elif score >= 3.0:
            return "양호"
        else:
            return "보통"


# ------------------------------ Node Factory ------------------------------

def _report_node_factory(cfg: ReportConfig):
    def node(state: ReportState) -> ReportState:
        company = _norm(((state.get("tech_summary") or {}).get("company")) or state.get("target_company", "")) or "(Unknown)"
        seg = _norm(state.get("segment", ""))
        ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        tech = state.get("tech_summary", {}) or {}
        market = state.get("market_eval", {}) or {}
        comp = state.get("competition", {}) or {}
        inv = state.get("investment", {}) or {}

        exec_draft = (
            f"투자 판단: **{inv.get('decision', '?')}** (총점 {inv.get('total_score', '?')}/100)\n"
            f"- 기술: {tech.get('tech_score', '-')} / 강점: {_fmt_list(tech.get('strengths', []))} / 약점: {_fmt_list(tech.get('weaknesses', []))}\n"
            f"- 시장: {market.get('market_score', '-')} (성장성), 리스크 안정성: {market.get('risk_score', '-')}\n"
            f"- 경쟁우위: {comp.get('edge_score', '-')} / 차별화: {comp.get('differentiation_summary', '')[:180]}\n"
            f"- 근거 요약: {inv.get('rationale', '')[:360]}"
        )

        llm = _get_llm(cfg.model, cfg.temperature) if cfg.use_llm_exec_summary else None
        exec_final = _refine_exec_summary(exec_draft, llm)

        sources = _sources_union(state, k=10)
        src_lines = "\n".join([f"- {s.get('title','')} — {s.get('url','')}" for s in sources]) or "- (출처 없음)"

        comp_lines = []
        for c in (comp.get("comparisons") or [])[:5]:
            name = c.get("name", "")
            pf = _norm(c.get("product_focus", ""))
            diff = _norm(c.get("tech_diff", ""))
            pos = _norm(c.get("positioning", ""))
            comp_lines.append(f"- **{name}**: {pf} / 차별화: {diff} / 포지셔닝: {pos}")
        comp_block = "\n".join(comp_lines) or "- (비교 데이터 없음)"

        md = f"""
# {company} 투자 분석 보고서

_작성시점: {ts}_

## 1) 요약 (Executive Summary)
{exec_final}

## 2) 기업 개요
- 대상: **{company}**
- 세그먼트 키워드: {seg or '(미지정)'}
- 최근 탐색 링크(일부):
{src_lines}

## 3) 기술 분석 요약
- 점수: **{tech.get('tech_score','-')} / 100**
- 강점: {_fmt_list(tech.get('strengths', []))}
- 약점: {_fmt_list(tech.get('weaknesses', []))}

> 핵심 기술 개요
{_norm(tech.get('summary',''))}

## 4) 시장 및 리스크
- 시장성 점수: **{market.get('market_score','-')} / 100**, 리스크 점수: **{market.get('risk_score','-')} / 100**
- 경쟁사(상위): {_fmt_list(market.get('competitors', []))}

> 시장 요약
{_norm(market.get('market_summary',''))}

> 리스크 요약
{_norm(market.get('risk_summary',''))}

## 5) 경쟁사 비교 스냅샷
- 경쟁우위(Edge) 점수: **{comp.get('edge_score','-')} / 100**

{comp_block}

## 6) 투자 판단
{_scores_table(inv.get('scores', {}), inv.get('weights', {}), inv.get('total_score', 0))}

**최종 판단:** {inv.get('decision','?')}

**근거:** {_norm(inv.get('rationale',''))}

## 7) 출처
{src_lines}
""".strip()

        filename = f"{cfg.filename_prefix}{company.replace(' ', '_')}.md"
        
        # PDF 생성 (옵션)
        pdf_path = None
        if cfg.generate_pdf:
            pdf_generator = PDFReportGenerator(output_dir=cfg.output_dir)
            pdf_path = pdf_generator.generate_pdf(state, company, evaluator="AI 투자분석팀")
        
        state["report"] = {
            "title": f"{company} 투자 분석 보고서", 
            "markdown": md, 
            "filename": filename,
            "pdf_path": pdf_path or ""
        }
        
        # 실행 결과 출력
        print("\n" + "="*80)
        print("✅ [6/6] 보고서 생성 완료")
        print("="*80)
        print(f"🏢 대상: {company}")
        print(f"📄 마크다운: {filename}")
        if pdf_path:
            print(f"📑 PDF: {pdf_path}")
        print(f"📊 보고서 길이: {len(md)} 자")
        print("="*80 + "\n")
        
        return state
    return node


# ------------------------------ Graph Builder ------------------------------

def build_report_graph(config: Optional[ReportConfig] = None):
    cfg = config or ReportConfig()
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph가 설치되어 있지 않습니다. `pip install langgraph` 후 다시 시도하세요.")
    g = StateGraph(ReportState)
    g.add_node("report", _report_node_factory(cfg))
    g.add_edge(START, "report")
    g.add_edge("report", END)
    return g.compile()


# ------------------------------ Helper ------------------------------

def run_report_generator(state: Dict[str, Any], config: Optional[ReportConfig] = None) -> Dict[str, Any]:
    app = build_report_graph(config)
    return app.invoke(state)

def save_report_markdown(report_text: str, output_dir: str = "outputs", filename: str = "investment_report.md"):
    """📝 보고서 결과를 Markdown 파일로 저장하는 함수"""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"✅ 마크다운 보고서가 저장되었습니다: {file_path}")
    return file_path

def save_report_complete(state: Dict[str, Any], output_dir: str = "outputs") -> Dict[str, str]:
    """
    📝 마크다운 + PDF 보고서 완전 저장
    
    Returns: {"markdown_path": "...", "pdf_path": "..."}
    """
    report = state.get("report", {})
    markdown = report.get("markdown", "")
    filename = report.get("filename", "investment_report.md")
    pdf_path = report.get("pdf_path", "")
    
    # 마크다운 저장
    md_path = save_report_markdown(markdown, output_dir, filename)
    
    result = {"markdown_path": md_path}
    
    if pdf_path:
        result["pdf_path"] = pdf_path
        print(f"✅ PDF 보고서: {pdf_path}")
    
    return result


# ------------------------------ CLI Test ------------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("📋 Report Generator 테스트 (Markdown + PDF)")
    print("=" * 80)
    
    dummy: ReportState = {
        "segment": "fintech ai fraud detection",
        "startup_search": {"docs": [{"title": "Zest AI overview", "url": "https://ex.com/zest"}]},
        "tech_summary": {"company": "Zest AI", "summary": "ML기반 신용평가.", "strengths": ["실시간 분석", "고정확도"], "weaknesses":["설명가능성"], "tech_score": 82},
        "market_eval": {"market_summary": "사기탐지 시장 성장.", "competitors": ["Featurespace", "FICO"], "risk_summary": "보안 리스크", "market_score": 79, "risk_score": 73},
        "competition": {"comparisons": [{"name":"Featurespace","product_focus":"실시간 사기탐지","tech_diff":"행동 프로파일링","positioning":"엔터프라이즈"}], "edge_score": 76},
        "investment": {"total_score": 78.5, "weights": {"tech":0.3,"market":0.25,"risk":0.2,"edge":0.25}, "scores": {"tech":82,"market":79,"risk":73,"edge":76}, "decision":"REVIEW", "rationale":"기술과 시장성은 우수하나 리스크 관리 개선 필요."}
    }
    
    # 마크다운 + PDF 생성
    config = ReportConfig(generate_pdf=True, output_dir="outputs")
    result = run_report_generator(dummy, config)
    
    print("\n✅ 보고서 생성 완료!")
    print(f"📄 마크다운: {result['report']['filename']}")
    if result['report'].get('pdf_path'):
        print(f"📑 PDF: {result['report']['pdf_path']}")
    
    print("\n📝 마크다운 미리보기:")
    print(result["report"]["markdown"][:600])
    print("\n...")
    
    # 파일 저장
    saved_paths = save_report_complete(result)
    print(f"\n💾 저장 완료: {saved_paths}")
