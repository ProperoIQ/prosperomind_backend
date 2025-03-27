# app/routers/dashboardlink_router.py
from fastapi import APIRouter, HTTPException
from typing import List
from app.repositories.dashboardlinks_repository import DashboardLinkRepository
from app.models.dashboardlink import DashboardLink, DashboardLinkCreate

router = APIRouter(prefix="/dashboardlinks", tags=["Dashboard Links"])

dashboardlink_repository = DashboardLinkRepository()

@router.post("/", response_model=DashboardLink)
def create_dashboard_link_route(dashboard_link_create: DashboardLinkCreate):
    new_dashboard_link = dashboardlink_repository.create_dashboard_link(dashboard_link_create)
    return new_dashboard_link

@router.get("/{board_id}", response_model=List[DashboardLink])
def get_dashboard_links_for_board_route(board_id: int):
    dashboard_link = dashboardlink_repository.get_dashboard_links_by_board(board_id)
    return dashboard_link

@router.put("/boards/{board_id}", response_model=List[DashboardLink])
def update_dashboard_links_for_board_route(board_id: int, dashboard_link: DashboardLinkCreate):
    updated_dashboard_links = dashboardlink_repository.update_dashboard_links_for_board(board_id, dashboard_link)
    if not updated_dashboard_links:
        raise HTTPException(status_code=404, detail="Dashboard links for the given board not found")
    return updated_dashboard_links

@router.delete("/boards/{board_id}", response_model=List[DashboardLink])
def delete_dashboard_links_for_board_route(board_id: int):
    deleted_dashboard_links = dashboardlink_repository.delete_dashboard_links_for_board(board_id)
    if not deleted_dashboard_links:
        raise HTTPException(status_code=404, detail="Dashboard links for the given board not found")
    return deleted_dashboard_links
