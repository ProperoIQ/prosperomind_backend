# app/models/dashboardlink.py
from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class DashboardLinkBase(BaseModel):
    board_id: int
    link: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_deleted: bool = False

class DashboardLinkCreate(DashboardLinkBase):
    pass

class DashboardLink(DashboardLinkBase):
    id: Optional[int] = None
    user_name: Optional[str] = None  # If you need this field

    class Config:
        orm_mode = True
