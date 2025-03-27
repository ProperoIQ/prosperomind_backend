# app/models/user_access.py
from typing import Optional, Any
from pydantic import BaseModel

class UserAccess(BaseModel):
    id: Optional[Any] = None
    client_user_id: int
    main_boards_access: str
    bcf_access: str
    boards_access: str
    prompts_access: str
    dashboardlinks_access: str
