import html
import base64
import re
import urllib.error
import urllib.parse
import urllib.request
import logging
import time
from dataclasses import dataclass

from core.config import settings

logger = logging.getLogger("app.research")


@dataclass
class ResearchResult:
    title: str
    url: str
    snippet: str


class WebResearchClient:
    """轻量联网研究客户端（多源降级：Bing → DuckDuckGo → 空结果）"""

    SEARCH_URL = "https://duckduckgo.com/html/"
    BING_URL = "https://www.bing.com/search"
    RELEVANT_TERMS = [
        "高考", "招生", "录取", "分数线", "投档", "位次", "教育考试院", "阳光高考",
        "就业质量", "毕业生", "薪资", "专业", "大学", "学院", "edu.cn", "gov.cn",
    ]
    TRUSTED_DOMAIN_HINTS = [
        "edu.cn",
        "gov.cn",
        "chsi.com.cn",
        "moe.gov.cn",
        "stats.gov.cn",
        "ncss.cn",
        "eol.cn",
        "gaokao.cn",
        "zs.",
        "zsb.",
        "sdzk.cn",
        "shmeea.edu.cn",
        "jseea.cn",
        "zjzs.net",
        "eea.gd.gov.cn",
        "bjeea.cn",
        "zhaokao.net",
        "hebeea.edu.cn",
        "sxkszx.cn",
        "nm.zsks.cn",
        "lnzsks.com",
        "jleea.edu.cn",
        "lzk.hl.cn",
        "ahzsks.cn",
        "eeafj.cn",
        "jxeea.cn",
        "haeea.cn",
        "hbea.edu.cn",
        "hneeb.cn",
        "gxeea.cn",
        "ea.hainan.gov.cn",
        "cqksy.cn",
        "sceea.cn",
        "zsksy.guizhou.gov.cn",
        "ynzs.cn",
        "zsks.edu.xizang.gov.cn",
        "sneea.cn",
        "ganseea.cn",
        "qhjyks.com",
        "nxjyks.cn",
        "xjzk.gov.cn",
    ]
    BLOCKED_DOMAIN_HINTS = [
        "wenku.baidu.com",
        "zhidao.baidu.com",
        "baijiahao.baidu.com",
        "docin.com",
        "zhihu.com",
        "csdn.net",
        "sohu.com",
        "163.com",
        "toutiao.com",
        "bilibili.com",
        "douyin.com",
        "xiaohongshu.com",
    ]

    def __init__(self):
        self.timeout = settings.research_timeout

    def search(self, query: str, limit: int = 4) -> list[ResearchResult]:
        """搜索：先尝试Bing，失败则降级到DuckDuckGo"""
        try:
            results = self._search_bing(query, limit)
            if results:
                logger.info(f"Bing returned {len(results)} results for: {query[:30]}...")
                return results
        except Exception as e:
            logger.warning(f"Bing search failed: {e}")

        try:
            results = self._search_duckduckgo(query, limit)
            if results:
                logger.info(f"DuckDuckGo returned {len(results)} results for: {query[:30]}...")
                return results
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")

        logger.error(f"All search sources failed for: {query[:30]}...")
        return []

    def research(
        self,
        queries: list[str],
        limit_per_query: int = 3,
        max_results: int = 6,
        max_queries: int = 8,
        max_seconds: float | None = None,
    ) -> list[ResearchResult]:
        seen = set()
        results: list[ResearchResult] = []
        started = time.monotonic()
        for query in queries[:max_queries]:
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                logger.warning(f"Research budget exhausted after {len(results)} results")
                break
            for item in self.search(query, limit=limit_per_query):
                key = item.url.rstrip("/")
                if key in seen:
                    continue
                seen.add(key)
                results.append(item)
                if len(results) >= max_results:
                    return self._sort_by_source_quality(results)
        return self._sort_by_source_quality(results)

    def build_summary(self, results: list[ResearchResult]) -> str:
        if not results:
            return "联网研究未拿到可用结果；回答时必须明确说明数据来源不足，只能基于本地数据和原则做初步判断。"

        lines = ["联网研究摘要："]
        for index, item in enumerate(results, start=1):
            snippet = item.snippet or "无摘要"
            lines.append(f"{index}. {item.title} - {snippet}（来源：{item.url}）")
        return "\n".join(lines)

    def _search_duckduckgo(self, query: str, limit: int) -> list[ResearchResult]:
        params = urllib.parse.urlencode({"q": query})
        request = urllib.request.Request(
            f"{self.SEARCH_URL}?{params}",
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                page = response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            logger.warning(f"DuckDuckGo request failed: {e}")
            return []

        return self._parse_duckduckgo(page, limit=limit)

    def _search_bing(self, query: str, limit: int) -> list[ResearchResult]:
        params = urllib.parse.urlencode({"q": query, "setlang": "zh-CN"})
        request = urllib.request.Request(
            f"{self.BING_URL}?{params}",
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                page = response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            logger.warning(f"Bing request failed: {e}")
            return []

        return self._parse_bing(page, limit=limit)

    def _parse_duckduckgo(self, page: str, limit: int) -> list[ResearchResult]:
        results: list[ResearchResult] = []
        blocks = re.findall(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            page,
            flags=re.S,
        )

        if not blocks:
            blocks = re.findall(
                r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                page,
                flags=re.S,
            )
            blocks = [(url, title, "") for url, title in blocks]

        for raw_url, raw_title, raw_snippet in blocks[:limit]:
            title = self._strip_html(raw_title)
            snippet = self._strip_html(raw_snippet)
            url = self._clean_url(raw_url)
            if title and url and self._is_relevant(title, url, snippet):
                results.append(ResearchResult(title=title, url=url, snippet=snippet))
        return results

    def _parse_bing(self, page: str, limit: int) -> list[ResearchResult]:
        results: list[ResearchResult] = []
        blocks = re.findall(r'<li class="b_algo".*?</li>', page, flags=re.S)
        for block in blocks:
            title_match = re.search(r"<h2[^>]*>\s*<a[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>", block, flags=re.S)
            if not title_match:
                continue
            snippet_match = re.search(r"<p[^>]*>(.*?)</p>", block, flags=re.S)
            url = self._clean_url(title_match.group(1))
            title = self._strip_html(title_match.group(2))
            snippet = self._strip_html(snippet_match.group(1) if snippet_match else "")
            if title and url and self._is_relevant(title, url, snippet):
                results.append(ResearchResult(title=title, url=url, snippet=snippet))
            if len(results) >= limit:
                break
        return results

    def _is_relevant(self, title: str, url: str, snippet: str) -> bool:
        haystack = f"{title} {url} {snippet}".lower()
        if any(domain in haystack for domain in self.BLOCKED_DOMAIN_HINTS):
            return False
        if any(domain in haystack for domain in self.TRUSTED_DOMAIN_HINTS):
            return True
        return False

    def _sort_by_source_quality(self, results: list[ResearchResult]) -> list[ResearchResult]:
        def score(item: ResearchResult) -> int:
            url = item.url.lower()
            if any(domain in url for domain in ["gov.cn", "moe.gov.cn", "stats.gov.cn"]):
                return 0
            if any(domain in url for domain in ["edu.cn", "chsi.com.cn", "ncss.cn"]):
                return 1
            if any(domain in url for domain in self.TRUSTED_DOMAIN_HINTS):
                return 2
            return 3

        return sorted(results, key=score)

    def _clean_url(self, url: str) -> str:
        url = html.unescape(url)
        if "bing.com/ck/a" in url:
            parsed = urllib.parse.urlparse(url)
            query = urllib.parse.parse_qs(parsed.query)
            encoded = (query.get("u") or [""])[0]
            if encoded:
                try:
                    if encoded.startswith("a1"):
                        encoded = encoded[2:]
                    padding = "=" * (-len(encoded) % 4)
                    decoded = base64.urlsafe_b64decode(encoded + padding).decode("utf-8", errors="replace")
                    if decoded.startswith(("http://", "https://")):
                        return decoded
                except (ValueError, OSError):
                    pass
        if url.startswith("//duckduckgo.com/l/?"):
            parsed = urllib.parse.urlparse("https:" + url)
            query = urllib.parse.parse_qs(parsed.query)
            if query.get("uddg"):
                return query["uddg"][0]
        return url

    def _strip_html(self, value: str) -> str:
        value = re.sub(r"<.*?>", "", value or "")
        value = html.unescape(value)
        return re.sub(r"\s+", " ", value).strip()


web_research_client = WebResearchClient()
