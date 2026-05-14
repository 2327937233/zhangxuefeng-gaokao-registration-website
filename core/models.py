from typing import Optional, List
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class UserProfile(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    score: Optional[int] = None
    rank: Optional[int] = None
    province: Optional[str] = None
    subjects: Optional[str] = None
    family_background: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("family_background", "familyBackground", "familyCondition"),
        description="如: 普通家庭/中产/富裕",
    )
    city_preference: Optional[List[str]] = Field(
        default=None,
        validation_alias=AliasChoices("city_preference", "cityPreference", "targetCities", "cityPref"),
        description="如: ['一线', '新一线']",
    )
    major_preference: Optional[List[str]] = Field(
        default=None,
        validation_alias=AliasChoices("major_preference", "majorPreference", "majorDirection", "majorPref"),
        description="如: ['计算机', '电子信息']",
    )
    risk_appetite: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("risk_appetite", "riskAppetite", "riskPreference", "risk"),
        description="稳妥/均衡/激进",
    )
    industry_acceptance: Optional[List[str]] = None
    willing_grad_school: Optional[bool] = None

    @field_validator("city_preference", "major_preference", mode="before")
    @classmethod
    def _split_preference_text(cls, value):
        if value is None or isinstance(value, list):
            return value
        if isinstance(value, str):
            pieces = [item.strip() for item in value.replace("，", "、").replace(",", "、").split("、")]
            return [item for item in pieces if item]
        return value


class OptionItem(BaseModel):
    type: str = Field(..., description="major 或 school")
    name: str
    school: Optional[str] = None


class ConsultRequest(BaseModel):
    question: str
    context: Optional[UserProfile] = None
    session_id: Optional[str] = Field(default=None, description="关联的会话ID，用于多轮对话")


class ThinkingStep(BaseModel):
    step: str
    analysis: str


class ConsultRecommendationPlan(BaseModel):
    order: int
    risk_level: str
    school: str
    major: str
    overview: str
    recommendation_reason: str
    probability: int
    median_salary_5yr: Optional[int] = None
    median_salary_display: str = "待核验"
    irreplaceability: Optional[int] = None
    probability_basis: str = "模拟估计值：基于当前画像、冲稳保规则和待核验投档位次粗排"
    salary_basis: str = "模拟估计值：基于本地专业库和就业质量报告检索口径估算"
    data_basis: str = "规则模拟/本地估算，仅供排序参考"
    citations: List[str] = []


class ConsultResponse(BaseModel):
    answer: str
    thinking_process: List[ThinkingStep]
    follow_up_questions: List[str]
    confidence: str  # low / medium / high
    citations: List[str] = []
    recommendation_plans: List[ConsultRecommendationPlan] = []


class EvaluateRequest(BaseModel):
    options: List[OptionItem]
    user_profile: UserProfile


class HeuristicScore(BaseModel):
    total: int
    employment_reversal: int
    social_sieve: int
    irreplaceability: int
    median_principle: int
    family_background: int
    city_priority: int
    fortune500_test: int
    ten_year_test: int


class EvaluationResult(BaseModel):
    recommendation: str
    scores: dict
    analysis: str
    red_flags: List[str]
    uncertainties: List[str]


class SoulQuestion(BaseModel):
    field: str
    question: str
    priority: str  # critical / high / medium / low


class SoulQuestionsRequest(BaseModel):
    known_info: dict


class SoulQuestionsResponse(BaseModel):
    questions: List[SoulQuestion]
    missing_critical: List[str]


class QuoteResponse(BaseModel):
    quote: str
    source: str
    scene: str = ""
    topics: list[str] = []
    is_classic: bool = False


class MajorItem(BaseModel):
    id: str
    name: str
    category: str
    employment_rate: float
    salary_median_5yr: int
    salary_entry: int
    requires_grad_school: bool
    irreplaceability: int
    tags: List[str]
    description: str
    risk_factors: List[str]
    data_source: str = Field(default="基于行业经验的估算值，非官方统计，仅供参考", description="数据来源说明")
    data_reliability: str = Field(default="estimate", description="数据可靠性等级: estimate/simulated/reported")


class SchoolItem(BaseModel):
    id: str
    name: str
    province: Optional[str] = None
    official_url: Optional[str] = None
    level: str
    city: str
    tier: str
    type: str
    employment_rate: float
    average_salary: int
    tags: List[str]
    data_source: str = Field(default="基于行业经验的估算值，非官方统计，仅供参考", description="数据来源说明")
    data_reliability: str = Field(default="estimate", description="数据可靠性等级: estimate/simulated/reported")


# ===================== Agent Models =====================

class UserPreferences(BaseModel):
    province: str = Field(..., description="考生省份")
    score: int = Field(..., description="高考分数")
    rank: Optional[int] = Field(None, description="全省位次")
    subjects: Optional[str] = Field(None, description="选科组合，如：物化生")
    family_background: Optional[str] = Field("普通家庭", description="家庭条件")
    city_preference: Optional[List[str]] = Field(None, description="意向城市列表")
    major_preference: Optional[List[str]] = Field(None, description="意向专业方向")
    risk_appetite: Optional[str] = Field("均衡", description="风险偏好：稳妥/均衡/激进")
    willing_grad_school: Optional[bool] = Field(None, description="是否接受深造")
    industry_avoid: Optional[List[str]] = Field(None, description="绝对不接受的行业")
    allow_military_schools: bool = Field(False, description="是否允许推荐军校/部队院校")


class PlanOption(BaseModel):
    order: int = Field(..., description="志愿顺序")
    school: str
    major: str
    major_group: Optional[str] = Field(None, description="专业组")
    risk_level: str = Field(..., description="冲/稳/保")
    probability: int = Field(..., ge=0, le=100, description="录取概率")
    median_salary_5yr: Optional[int] = None
    irreplaceability: Optional[int] = None
    reason: str = Field(default="", description="推荐理由（就业倒推）")
    risk_warning: Optional[str] = None
    tags: List[str] = []
    fortune500_pass: bool = False


class RecommendRequest(BaseModel):
    user: UserPreferences
    limit: int = Field(8, ge=3, le=20, description="返回志愿数量")


class RecommendResponse(BaseModel):
    plans: List[PlanOption]
    summary: str
    chong_count: int
    wen_count: int
    bao_count: int
    thinking_process: List[ThinkingStep]
    red_flags: List[str] = []


class CompareRequest(BaseModel):
    plans: List[PlanOption]
    user: UserPreferences


class CompareResult(BaseModel):
    best_choice: PlanOption
    comparison_table: str
    dimension_scores: dict
    final_verdict: str
    thinking_process: List[ThinkingStep]


class InsightRequest(BaseModel):
    target_type: str = Field(..., description="major / school / industry")
    target_name: str
    user: Optional[UserPreferences] = None


class InsightResponse(BaseModel):
    target: str
    target_type: str
    overview: str
    median_salary: Optional[int] = None
    employment_rate: Optional[float] = None
    irreplaceability: Optional[int] = None
    trend_analysis: str
    risk_factors: List[str] = []
    opportunities: List[str] = []
    similar_options: List[str] = []
    thinking_process: List[ThinkingStep]


class PressureTestRequest(BaseModel):
    plan: PlanOption
    user: UserPreferences
    compare_with: Optional[PlanOption] = None


class PressureTestResponse(BaseModel):
    scenario: str
    year_10_salary_median: Optional[int] = None
    year_10_salary_compare: Optional[int] = None
    analysis: str
    stress_conclusion: str
    acceptable: bool
    thinking_process: List[ThinkingStep]


class AnalyzeRequest(BaseModel):
    target_type: str = Field(..., description="major / school")
    target_name: str
    school_name: Optional[str] = None
    user: Optional[UserPreferences] = None


class AnalyzeResponse(BaseModel):
    target: str
    target_type: str
    deep_analysis: str
    eight_dimensions: dict
    suitability_score: int = Field(..., ge=0, le=100)
    for_whom: str
    against_whom: str
    thinking_process: List[ThinkingStep]
    zxf_quote: str


# ===================== Session Models =====================

class SessionMessage(BaseModel):
    role: str = Field(..., description="user / assistant / system")
    content: str
    timestamp: str = Field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())


class Session(BaseModel):
    id: str
    title: str
    user_profile: Optional[UserProfile] = None
    messages: List[SessionMessage] = []
    created_at: str = Field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())


class SessionCreateRequest(BaseModel):
    title: Optional[str] = Field(default=None, description="会话标题，默认自动生成")
    user_profile: Optional[UserProfile] = None


class SessionUpdateProfileRequest(BaseModel):
    user_profile: Optional[UserProfile] = None


class SessionRenameRequest(BaseModel):
    title: str
