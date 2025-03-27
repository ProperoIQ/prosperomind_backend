# app/routers/master_settings_router.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.repositories.master_settings_repository import MasterSettingsRepository
from app.models.master_settings import MasterSettings

router = APIRouter(prefix="/master-settings", tags=["Master Settings"])

master_settings_repository = MasterSettingsRepository()

@router.post("/", response_model=MasterSettings)
async def create_master_settings(master_settings: MasterSettings):
    try:
        # Attempt to create Master Settings
        created_settings = master_settings_repository.create_master_settings(master_settings)
        return created_settings
    except Exception as e:
        # Check if the error is due to a duplicate key violation
        if "duplicate key value violates unique constraint" in str(e):
            raise HTTPException(status_code=400, detail="Master Settings with this name already exist")
        else:
            # Re-raise the exception if it's not related to duplicate key violation
            raise HTTPException(status_code=500, detail="Master Settings API failed to process")

@router.get("/", response_model=List[MasterSettings])
async def get_all_master_settings():
    all_settings = master_settings_repository.get_all_master_settings()
    return all_settings

@router.get("/{settings_id}", response_model=MasterSettings)
async def get_master_settings(settings_id: int):
    settings = master_settings_repository.get_master_settings(settings_id)
    if not settings:
        raise HTTPException(status_code=404, detail="Master Settings not found")
    return settings

@router.put("/{settings_id}", response_model=MasterSettings)
async def update_master_settings(settings_id: int, master_settings: MasterSettings):
    updated_settings = master_settings_repository.update_master_settings(settings_id, master_settings)
    if not updated_settings:
        raise HTTPException(status_code=404, detail="Master Settings not found")
    return updated_settings

@router.delete("/{settings_id}", response_model=dict)
async def delete_master_settings(settings_id: int):
    deleted_settings = master_settings_repository.delete_master_settings(settings_id)
    if not deleted_settings:
        raise HTTPException(status_code=404, detail="Master Settings not found")
    response_data = {"status_code": 200, "detail": "Master Settings deleted successfully"}
    return response_data
