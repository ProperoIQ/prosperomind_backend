from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.models.profitability_dashboard_settings import ProfitabilityDashboardSettings
from app.repositories.profitability_dashboard_settings_repository import ProfitabilityDashboardSettingsRepository

router = APIRouter()
repo = ProfitabilityDashboardSettingsRepository()

@router.post("/dashboard_settings/", response_model=ProfitabilityDashboardSettings)
async def create_profitability_dashboard_settings(settings: ProfitabilityDashboardSettings) -> ProfitabilityDashboardSettings:
    return repo.create_profitability_dashboard_settings(settings)

@router.get("/dashboard_settings/{settings_id}", response_model=ProfitabilityDashboardSettings)
async def get_profitability_dashboard_settings(settings_id: int) -> ProfitabilityDashboardSettings:
    settings = repo.get_profitability_dashboard_settings(settings_id)
    if not settings:
        raise HTTPException(status_code=404, detail="Profitability dashboard settings not found")
    return settings

@router.put("/dashboard_settings/{settings_id}", response_model=ProfitabilityDashboardSettings)
async def update_profitability_dashboard_settings(settings_id: int, settings: ProfitabilityDashboardSettings) -> ProfitabilityDashboardSettings:
    existing_settings = repo.get_profitability_dashboard_settings(settings_id)
    if not existing_settings:
        raise HTTPException(status_code=404, detail="Profitability dashboard settings not found")
    return repo.update_profitability_dashboard_settings(settings_id, settings)

@router.delete("/dashboard_settings/{settings_id}", response_model=ProfitabilityDashboardSettings)
async def delete_profitability_dashboard_settings(settings_id: int) -> ProfitabilityDashboardSettings:
    existing_settings = repo.get_profitability_dashboard_settings(settings_id)
    if not existing_settings:
        raise HTTPException(status_code=404, detail="Profitability dashboard settings not found")
    return repo.delete_profitability_dashboard_settings(settings_id)

@router.get("/dashboard_settings/board/{board_id}", response_model=ProfitabilityDashboardSettings)
async def get_profitability_dashboard_settings_by_board_id(board_id: int) -> List[ProfitabilityDashboardSettings]:
    settings_list = repo.get_profitability_dashboard_settings_by_board_id(board_id)
    if not settings_list:
        raise HTTPException(status_code=404, detail="No profitability dashboard settings found for the board")
    return settings_list
