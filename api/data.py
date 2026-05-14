import random
from fastapi import APIRouter, Query

from core.models import QuoteResponse, MajorItem, SchoolItem
from data import majors, schools, quotes

router = APIRouter(prefix="/data", tags=["data"])


@router.get("/quote", response_model=QuoteResponse)
async def get_random_quote(
    scene: str = Query("", description="按场景筛选：直播/讲座/演讲/采访/朋友圈/口头禅"),
    topic: str = Query("", description="按主题筛选，如：专业选择/家庭条件/城市选择"),
    classic_only: bool = Query(False, description="仅返回经典语录"),
):
    """随机获取一条张雪峰金句，支持按场景、主题和是否经典筛选"""
    pool = quotes
    if scene:
        pool = [q for q in pool if q.get("scene") == scene]
    if topic:
        pool = [q for q in pool if topic in q.get("topics", [])]
    if classic_only:
        pool = [q for q in pool if q.get("is_classic")]
    if not pool:
        return QuoteResponse(
            quote="我跟你说，这个事我还真不太了解，但按我的经验——先别急。",
            source="系统默认",
            scene="口头禅",
            topics=["兜底回复"],
            is_classic=True,
        )
    q = random.choice(pool)
    return QuoteResponse(
        quote=q["text"],
        source=q["source"],
        scene=q.get("scene", ""),
        topics=q.get("topics", []),
        is_classic=q.get("is_classic", False),
    )


@router.get("/quotes/scenes")
async def list_quote_scenes() -> list[str]:
    """获取所有语录场景分类"""
    scenes = sorted({q.get("scene", "") for q in quotes if q.get("scene")})
    return scenes


@router.get("/quotes/topics")
async def list_quote_topics() -> list[str]:
    """获取所有语录主题标签"""
    topics = set()
    for q in quotes:
        topics.update(q.get("topics", []))
    return sorted(topics)


@router.get("/quotes")
async def list_quotes(
    scene: str = Query("", description="按场景筛选"),
    topic: str = Query("", description="按主题筛选"),
    classic_only: bool = Query(False, description="仅返回经典语录"),
) -> list[QuoteResponse]:
    """获取语录列表，支持筛选"""
    result = quotes
    if scene:
        result = [q for q in result if q.get("scene") == scene]
    if topic:
        result = [q for q in result if topic in q.get("topics", [])]
    if classic_only:
        result = [q for q in result if q.get("is_classic")]
    return [
        QuoteResponse(
            quote=q["text"],
            source=q["source"],
            scene=q.get("scene", ""),
            topics=q.get("topics", []),
            is_classic=q.get("is_classic", False),
        )
        for q in result
    ]


@router.get("/majors")
async def list_majors(
    keyword: str = Query("", description="关键词搜索"),
    category: str = Query("", description="学科门类"),
) -> list[MajorItem]:
    """获取专业列表"""
    result = []
    for m in majors:
        if keyword and keyword not in m["name"]:
            continue
        if category and m["category"] != category:
            continue
        result.append(MajorItem(**m))
    return result


@router.get("/schools")
async def list_schools(
    province: str = Query("", description="省份/地区"),
    city: str = Query("", description="城市"),
    level: str = Query("", description="层次：985/211/双一流/普通一本/普通二本"),
    tier: str = Query("", description="城市级别：一线/新一线/二线/三线/四线"),
) -> list[SchoolItem]:
    """获取院校列表"""
    result = []
    for s in schools:
        if province and province not in s.get("province", ""):
            continue
        if city and city not in s["city"]:
            continue
        if level and s["level"] != level:
            continue
        if tier and s["tier"] != tier:
            continue
        result.append(SchoolItem(**s))
    return result
