from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List


# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
    is_admin: Optional[bool] = False


# User Schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr
    is_admin: bool = False


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# Trade Schemas
class TradeBase(BaseModel):
    symbol: str
    direction: str
    price: float
    lot_size: float
    model_used: str
    status: Optional[str] = "OPEN"
    mt5_order_ticket: Optional[int] = None


class TradeCreate(TradeBase):
    pass


class Trade(TradeBase):
    id: int
    user_id: int
    timestamp: datetime
    pnl: Optional[float] = None

    class Config:
        from_attributes = True


class PortfolioResponse(BaseModel):
    trades: List[Trade]
    total_trades: int
    win_rate: float
    total_pnl: float
    sharpe_ratio: float
    profit_factor: float
    max_drawdown: float
    equity_curve: List[dict] # {time: unix, value: cumulative_pnl}
    model_performance: dict # {model_name: {wins: int, total: int, pnl: float}}
