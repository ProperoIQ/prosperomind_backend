from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.models.cashflow_dashboard_settings import CashFlowDashboardSettings
from app.repositories.cashflow_dashboard_settings_repository import CashFlowDashboardSettingsRepository

router = APIRouter()
repo = CashFlowDashboardSettingsRepository()

@router.post("/cashflow_dashboard_settings/", response_model=CashFlowDashboardSettings)
async def create_cashflow_dashboard_settings(settings: CashFlowDashboardSettings) -> CashFlowDashboardSettings:
    return repo.create_cashflow_dashboard_settings(settings)

@router.get("/cashflow_dashboard_settings/{settings_id}", response_model=CashFlowDashboardSettings)
async def get_cashflow_dashboard_settings(settings_id: int) -> CashFlowDashboardSettings:
    settings = repo.get_cashflow_dashboard_settings(settings_id)
    if not settings:
        raise HTTPException(status_code=404, detail="Cash flow dashboard settings not found")
    return settings

@router.put("/cashflow_dashboard_settings/{settings_id}", response_model=CashFlowDashboardSettings)
async def update_cashflow_dashboard_settings(settings_id: int, settings: CashFlowDashboardSettings) -> CashFlowDashboardSettings:
    existing_settings = repo.get_cashflow_dashboard_settings(settings_id)
    if not existing_settings:
        raise HTTPException(status_code=404, detail="Cash flow dashboard settings not found")
    return repo.update_cashflow_dashboard_settings(settings_id, settings)

@router.delete("/cashflow_dashboard_settings/{settings_id}", response_model=CashFlowDashboardSettings)
async def delete_cashflow_dashboard_settings(settings_id: int) -> CashFlowDashboardSettings:
    existing_settings = repo.get_cashflow_dashboard_settings(settings_id)
    if not existing_settings:
        raise HTTPException(status_code=404, detail="Cash flow dashboard settings not found")
    return repo.delete_cashflow_dashboard_settings(settings_id)

@router.get("/cashflow_dashboard_settings/board/{board_id}", response_model=List[CashFlowDashboardSettings])
async def get_cashflow_dashboard_settings_by_board_id(board_id: int) -> List[CashFlowDashboardSettings]:
    settings_list = repo.get_cashflow_dashboard_settings_by_board_id(board_id)
    if not settings_list:
        raise HTTPException(status_code=404, detail="No cash flow dashboard settings found for the board")
    return settings_list
