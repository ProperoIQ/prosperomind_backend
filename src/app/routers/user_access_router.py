# app/routers/user_access_router.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.repositories.user_access_repository import UserAccessRepository
from app.models.user_access import UserAccess

router = APIRouter(prefix="/user-access", tags=["User Access"])

user_access_repository = UserAccessRepository()

@router.post("/", response_model=UserAccess)
async def create_user_access(user_access: UserAccess):
    created_access = user_access_repository.create_user_access(user_access)
    return created_access

@router.get("/", response_model=List[UserAccess])
async def get_all_user_access():
    all_access = user_access_repository.get_all_user_access()
    return all_access

@router.get("/{access_id}", response_model=UserAccess)
async def get_user_access(access_id: int):
    access = user_access_repository.get_user_access(access_id)
    if not access:
        raise HTTPException(status_code=404, detail="User Access not found")
    return access

@router.put("/{access_id}", response_model=UserAccess)
async def update_user_access(access_id: int, user_access: UserAccess):
    updated_access = user_access_repository.update_user_access(access_id, user_access)
    if not updated_access:
        raise HTTPException(status_code=404, detail="User Access not found")
    return updated_access

@router.delete("/{access_id}", response_model=dict)
async def delete_user_access(access_id: int):
    deleted_access = user_access_repository.delete_user_access(access_id)
    if not deleted_access:
        raise HTTPException(status_code=404, detail="User Access not found")
    response_data = {"status_code": 200, "detail": "User Access deleted successfully"}
    return response_data
