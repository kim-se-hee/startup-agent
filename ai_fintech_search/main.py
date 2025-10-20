# ai_fintech_search/main.py
import argparse
import json
from ai_fintech_search.agents.startup_search_agent import run_startup_search
from ai_fintech_search import config

def main():
    # 선택: 여기서도 재확인 가능
    config.ensure_required_env()

    p = argparse.ArgumentParser(description="AI Fintech Startup Finder (2025)")
    p.add_argument("--region", default="Global", help="Global | Korea")
    p.add_argument("--limit", type=int, default=10)
    args = p.parse_args()

    hits = run_startup_search(region=args.region, limit=args.limit, language="ko")
    print(json.dumps({"items": [h.model_dump() for h in hits]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
