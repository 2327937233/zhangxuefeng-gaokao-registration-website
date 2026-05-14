from fastapi import APIRouter

from core.models import (
    RecommendRequest, RecommendResponse,
    CompareRequest, CompareResult,
    InsightRequest, InsightResponse,
    PressureTestRequest, PressureTestResponse,
    AnalyzeRequest, AnalyzeResponse,
)
from core.agent_engine import agent_engine
from middleware.error_handler import ValidationError

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """智能志愿推荐 Agent"""
    if not request.user or not request.user.province or not request.user.score:
        raise ValidationError("province 和 score 为必填项")
    return agent_engine.recommend(request)


@router.post("/compare", response_model=CompareResult)
async def compare(request: CompareRequest):
    """方案对比 Agent"""
    if not request.plans:
        raise ValidationError("plans 不能为空")
    return agent_engine.compare(request)


@router.post("/insights", response_model=InsightResponse)
async def insights(request: InsightRequest):
    """数据洞察 Agent"""
    if not request.target_name:
        raise ValidationError("target_name 不能为空")
    return agent_engine.insights(request)


@router.post("/pressure-test", response_model=PressureTestResponse)
async def pressure_test(request: PressureTestRequest):
    """10年后压迫测试 Agent"""
    if not request.plan:
        raise ValidationError("plan 不能为空")
    return agent_engine.pressure_test(request)


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """深度分析 Agent"""
    if not request.target_name:
        raise ValidationError("target_name 不能为空")
    return agent_engine.analyze(request)
