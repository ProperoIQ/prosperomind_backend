# app/routers/time_line_settings_router.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.repositories.time_line_settings_repository import TimeLineSettingsRepository
from app.models.time_line_settings import TimeLineSettings

router = APIRouter(prefix="/time-line-settings", tags=["Time Line Settings"])

time_line_settings_repository = TimeLineSettingsRepository()

@router.post("/", response_model=TimeLineSettings)
async def create_time_line_settings(time_line_settings: TimeLineSettings):
    try:
        # Attempt to create Time Line Settings
        created_settings = time_line_settings_repository.create_time_line_settings(time_line_settings)
        return created_settings
    except Exception as e:
        # Check if the error is due to a duplicate key violation
        if "duplicate key value violates unique constraint" in str(e):
            raise HTTPException(status_code=400, detail="Time Line Settings with this name already exist")
        else:
            # Re-raise the exception if it's not related to duplicate key violation
            raise HTTPException(status_code=500, detail="Time Line Settings API failed to process")

@router.get("/", response_model=List[TimeLineSettings])
async def get_all_time_line_settings():
    all_settings = time_line_settings_repository.get_all_time_line_settings()
    return all_settings

@router.get("/{settings_id}", response_model=TimeLineSettings)
async def get_time_line_settings(settings_id: int):
    settings = time_line_settings_repository.get_time_line_settings(settings_id)
    if not settings:
        raise HTTPException(status_code=404, detail="Time Line Settings not found")
    return settings

@router.put("/{settings_id}", response_model=TimeLineSettings)
async def update_time_line_settings(settings_id: int, time_line_settings: TimeLineSettings):
    updated_settings = time_line_settings_repository.update_time_line_settings(settings_id, time_line_settings)
    if not updated_settings:
        raise HTTPException(status_code=404, detail="Time Line Settings not found")
    return updated_settings

@router.delete("/{settings_id}", response_model=dict)
async def delete_time_line_settings(settings_id: int):
    deleted_settings = time_line_settings_repository.delete_time_line_settings(settings_id)
    if not deleted_settings:
        raise HTTPException(status_code=404, detail="Time Line Settings not found")
    response_data = {"status_code": 200, "detail": "Time Line Settings deleted successfully"}
    return response_data
