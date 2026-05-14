from fastapi import APIRouter

from core.models import (
    Session, SessionCreateRequest, SessionUpdateProfileRequest,
    SessionRenameRequest,
)
from core.session_manager import session_manager
from middleware.error_handler import NotFoundError

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=Session)
async def create_session(request: SessionCreateRequest):
    """创建新会话，可携带初始考生画像"""
    return session_manager.create_session(
        title=request.title,
        user_profile=request.user_profile,
    )


@router.get("", response_model=list[Session])
async def list_sessions():
    """获取所有会话列表，按更新时间倒序"""
    return session_manager.list_sessions()


@router.get("/{session_id}", response_model=Session)
async def get_session(session_id: str):
    """获取指定会话的详情（包含消息历史）"""
    session = session_manager.get_session(session_id)
    if not session:
        raise NotFoundError("会话不存在")
    return session


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话"""
    success = session_manager.delete_session(session_id)
    if not success:
        raise NotFoundError("会话不存在")
    return {"message": "会话已删除", "session_id": session_id}


@router.put("/{session_id}/profile", response_model=Session)
async def update_session_profile(session_id: str, request: SessionUpdateProfileRequest):
    """更新会话绑定的考生画像（切换不同考生画像）"""
    session = session_manager.update_profile(session_id, request.user_profile)
    if not session:
        raise NotFoundError("会话不存在")
    return session


@router.put("/{session_id}/rename", response_model=Session)
async def rename_session(session_id: str, request: SessionRenameRequest):
    """重命名会话"""
    session = session_manager.rename_session(session_id, request.title)
    if not session:
        raise NotFoundError("会话不存在")
    return session
