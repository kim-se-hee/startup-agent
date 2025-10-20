from urllib.parse import urlparse, urlunparse

def normalize_root_url(url: str) -> str:
    """
    쿼리스트링/프래그먼트 제거, 루트 경로로 정규화
    """
    try:
        u = urlparse(url)
        # path를 루트로 고정 (필요 시 유지하려면 수정)
        clean = u._replace(path="/", params="", query="", fragment="")
        return urlunparse(clean)
    except Exception:
        return url
