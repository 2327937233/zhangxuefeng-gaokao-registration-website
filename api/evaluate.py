from fastapi import APIRouter

from core.models import (
    EvaluateRequest, EvaluationResult,
    SoulQuestionsRequest, SoulQuestionsResponse,
)
from core.zxf_engine import evaluator
from middleware.error_handler import ValidationError

router = APIRouter(prefix="/evaluate", tags=["evaluate"])


@router.post("", response_model=EvaluationResult)
async def evaluate(request: EvaluateRequest):
    """
    决策评估引擎

    对一组选项（专业/院校）进行综合评估，输出8维度评分和推荐。
    """
    if not request.options:
        raise ValidationError("options 不能为空")
    return evaluator.evaluate(request.options, request.user_profile)


@router.post("/soul-questions", response_model=SoulQuestionsResponse)
async def soul_questions(request: SoulQuestionsRequest):
    """
    灵魂追问生成器

    根据已知信息，生成还需要追问的关键问题。
    """
    result = evaluator.generate_soul_questions(request.known_info)
    from core.models import SoulQuestion
    questions = [SoulQuestion(**q) for q in result["questions"]]
    return SoulQuestionsResponse(
        questions=questions,
        missing_critical=result["missing_critical"],
    )
