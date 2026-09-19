from sqlalchemy import Column, Integer, String, DateTime, Boolean
from app.core.database import Base
from datetime import datetime, timedelta


class PasswordReset(Base):
    __tablename__ = "password_resets"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, nullable=False)
    reset_code = Column(String(64), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_used = Column(Boolean, default=False)
    used_at = Column(DateTime, nullable=True)

    def is_expired(self) -> bool:
        return datetime.utcnow() > self.created_at + timedelta(minutes=15)
