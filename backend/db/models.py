from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from backend.db.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    trades = relationship("Trade", back_populates="user")


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String, index=True)
    direction = Column(String)
    price = Column(Float)
    lot_size = Column(Float)
    model_used = Column(String)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    pnl = Column(Float, nullable=True) # Simulated PnL
    
    # MT5 Sync Fields
    mt5_order_ticket = Column(Integer, nullable=True)
    status = Column(String, default="OPEN") # OPEN, CLOSED

    user = relationship("User", back_populates="trades")
