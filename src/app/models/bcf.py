# app/models/bcf.py
from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel

class BCF(BaseModel):
    id: Optional[Any] = None
    main_board_id: int
    name: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
