import uuid
import logging
from datetime import datetime
from typing import Optional

from core.models import Session, SessionMessage, UserProfile

logger = logging.getLogger("app.session")


class SessionManager:
    """内存会话管理器：支持多考生画像的会话隔离"""

    def __init__(self):
        # session_id -> Session
        self._sessions: dict[str, Session] = {}

    def create_session(self, title: Optional[str] = None, user_profile: Optional[UserProfile] = None) -> Session:
        """创建新会话"""
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        auto_title = title or f"咨询 {datetime.now().strftime('%m-%d %H:%M')}"
        session = Session(
            id=session_id,
            title=auto_title,
            user_profile=user_profile,
            messages=[],
        )
        self._sessions[session_id] = session
        logger.info(f"Session created: {session_id}, title={auto_title}")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[Session]:
        """列出所有会话，按更新时间倒序"""
        sessions = list(self._sessions.values())
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session deleted: {session_id}")
            return True
        return False

    def update_profile(self, session_id: str, user_profile: Optional[UserProfile]) -> Optional[Session]:
        """更新会话绑定的考生画像"""
        session = self._sessions.get(session_id)
        if not session:
            return None
        session.user_profile = user_profile
        session.updated_at = datetime.now().isoformat()
        logger.info(f"Session profile updated: {session_id}")
        return session

    def rename_session(self, session_id: str, title: str) -> Optional[Session]:
        """重命名会话"""
        session = self._sessions.get(session_id)
        if not session:
            return None
        session.title = title
        session.updated_at = datetime.now().isoformat()
        return session

    def add_message(self, session_id: str, role: str, content: str) -> Optional[Session]:
        """向会话添加消息"""
        session = self._sessions.get(session_id)
        if not session:
            return None
        message = SessionMessage(role=role, content=content)
        session.messages.append(message)
        session.updated_at = datetime.now().isoformat()

        # 限制历史消息数量，防止 token 过长（保留最近 20 条）
        if len(session.messages) > 20:
            session.messages = session.messages[-20:]

        return session

    def get_history_messages(self, session_id: str, limit: int = 10) -> list[dict]:
        """获取会话的历史消息，用于构建 LLM 多轮对话上下文"""
        session = self._sessions.get(session_id)
        if not session or not session.messages:
            return []

        # 取最近 limit 条，转换为 LLM 可用的格式
        recent = session.messages[-limit:]
        return [{"role": msg.role, "content": msg.content} for msg in recent]


session_manager = SessionManager()
