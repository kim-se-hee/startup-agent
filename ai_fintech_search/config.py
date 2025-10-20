# ai_fintech_search/config.py
import os
from dotenv import load_dotenv

# 1) .env 로드 (루트에 있는 .env 자동 탐지)
load_dotenv()

# 2) 환경변수 조회 (없으면 기본값)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

REQUIRED_ENV = ["OPENAI_API_KEY", "TAVILY_API_KEY"]

def ensure_required_env(required: list[str] = None) -> None:
    """
    필수 키가 모두 있는지 런타임 시점에 확인.
    - import 시점에 실패하지 않도록 '지연 검증'으로 둠
    """
    required = required or REQUIRED_ENV
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        lines = [f"- {k} 가 설정되지 않았습니다." for k in missing]
        msg = "다음 환경변수가 필요합니다:\n" + "\n".join(lines) + "\n(.env에 키를 설정하세요)"
        raise RuntimeError(msg)
