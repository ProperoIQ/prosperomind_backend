from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel
from sqlalchemy import text
from app.repositories.base_repository import BaseRepository
from app.models.cashflow_dashboard_settings import CashFlowDashboardSettings

class CashFlowDashboardSettingsRepository(BaseRepository):
    def __init__(self):
        super().__init__('CashFlowDashboardSettings')
        create_table_query = text("""
            CREATE TABLE IF NOT EXISTS CashFlowDashboardSettings (
                id SERIAL PRIMARY KEY,
                board_id INT REFERENCES Boards(id),
                name VARCHAR,
                tax FLOAT,
                financial_year_start VARCHAR,
                financial_year_end VARCHAR,
                forecast_period INT,
                output_response VARCHAR,
                publish_to_cfo BOOLEAN,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            );
        """)
        self.create_table(create_table_query)

    def create_cashflow_dashboard_settings(self, settings: CashFlowDashboardSettings) -> CashFlowDashboardSettings:
        query = text("""
            INSERT INTO CashFlowDashboardSettings (board_id, name, tax, financial_year_start, financial_year_end, 
                                                   forecast_period, output_response, publish_to_cfo, created_at, updated_at)
            VALUES (:board_id, :name, :tax, :financial_year_start, :financial_year_end, 
                    :forecast_period, :output_response, :publish_to_cfo, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING id, board_id, name, tax, financial_year_start, financial_year_end, 
                      forecast_period, output_response, publish_to_cfo, created_at, updated_at;
        """)
        values = settings.dict()
        cashflow_settings_data_tuple = self.execute_query(query, values)
        return CashFlowDashboardSettings(**dict(zip(CashFlowDashboardSettings.__annotations__, cashflow_settings_data_tuple)))

    def get_cashflow_dashboard_settings(self, settings_id: int) -> CashFlowDashboardSettings:
        query = text("""
            SELECT * FROM CashFlowDashboardSettings WHERE id = :settings_id;
        """)
        values = {"settings_id": settings_id}
        cashflow_settings_data_tuple = self.execute_query(query, values)
        if cashflow_settings_data_tuple is None:
            return {}
        return CashFlowDashboardSettings(**dict(zip(CashFlowDashboardSettings.__annotations__, cashflow_settings_data_tuple)))

    def update_cashflow_dashboard_settings(self, settings_id: int, settings: CashFlowDashboardSettings) -> CashFlowDashboardSettings:
        query = text("""
            UPDATE CashFlowDashboardSettings
            SET board_id = :board_id, name = :name, tax = :tax, financial_year_start = :financial_year_start, 
                financial_year_end = :financial_year_end, forecast_period = :forecast_period, 
                output_response = :output_response, publish_to_cfo = :publish_to_cfo, updated_at = CURRENT_TIMESTAMP
            WHERE id = :settings_id
            RETURNING id, board_id, name, tax, financial_year_start, financial_year_end, 
                      forecast_period, output_response, publish_to_cfo, created_at, updated_at;
        """)
        values = settings.dict()
        values["settings_id"] = settings_id
        cashflow_settings_data_tuple = self.execute_query(query, values)
        return CashFlowDashboardSettings(**dict(zip(CashFlowDashboardSettings.__annotations__, cashflow_settings_data_tuple)))

    def delete_cashflow_dashboard_settings(self, settings_id: int) -> CashFlowDashboardSettings:
        query = text("""
            DELETE FROM CashFlowDashboardSettings WHERE id = :settings_id
            RETURNING id, board_id, name, tax, financial_year_start, financial_year_end, 
                      forecast_period, output_response, publish_to_cfo, created_at, updated_at;
        """)
        values = {"settings_id": settings_id}
        cashflow_settings_data_tuple = self.execute_query(query, values)
        return CashFlowDashboardSettings(**dict(zip(CashFlowDashboardSettings.__annotations__, cashflow_settings_data_tuple)))

    def get_cashflow_dashboard_settings_by_board_id(self, board_id: int) -> List[CashFlowDashboardSettings]:
        query = text("""
            SELECT * FROM CashFlowDashboardSettings WHERE board_id = :board_id;
        """)
        values = {"board_id": board_id}
        cashflow_settings_data_list = self.execute_query_all(query, values)
        if cashflow_settings_data_list is None:
            return {}
        cashflow_settings_list = [CashFlowDashboardSettings(**dict(zip(CashFlowDashboardSettings.__annotations__, settings_data))) for settings_data in cashflow_settings_data_list]
        return cashflow_settings_list

