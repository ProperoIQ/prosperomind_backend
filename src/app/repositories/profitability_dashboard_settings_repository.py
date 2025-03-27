# app/repositories/profitability_dashboard_settings_repository.py
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.profitability_dashboard_settings import ProfitabilityDashboardSettings
from app.repositories.base_repository import BaseRepository

class ProfitabilityDashboardSettingsRepository(BaseRepository):
    def __init__(self):
        super().__init__('ProfitabilityDashboardSettings')
        create_table_query = text(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id SERIAL PRIMARY KEY,
                board_id INT REFERENCES Boards(id),
                name VARCHAR,
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

    def create_profitability_dashboard_settings(self, dashboard_settings: ProfitabilityDashboardSettings) -> ProfitabilityDashboardSettings:
        query = text(f"""
            INSERT INTO {self.table_name} (board_id, name, financial_year_start, financial_year_end, forecast_period, output_response, publish_to_cfo, created_at, updated_at)
            VALUES (:board_id, :name, :financial_year_start, :financial_year_end, :forecast_period, :output_response, :publish_to_cfo, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING id, board_id, name, financial_year_start, financial_year_end, forecast_period, output_response, publish_to_cfo, created_at, updated_at;
        """)
        values = dashboard_settings.dict()
        dashboard_settings_data_tuple = self.execute_query(query, values)
        return ProfitabilityDashboardSettings(**dict(zip(ProfitabilityDashboardSettings.__annotations__, dashboard_settings_data_tuple)))

    def get_profitability_dashboard_settings(self, dashboard_settings_id: int) -> ProfitabilityDashboardSettings:
        query = text(f"""
            SELECT * FROM {self.table_name} WHERE id = :dashboard_settings_id;
        """)
        values = {"dashboard_settings_id": dashboard_settings_id}
        dashboard_settings_data_tuple = self.execute_query(query, values)
        if dashboard_settings_data_tuple is None:
            return {}
        return ProfitabilityDashboardSettings(**dict(zip(ProfitabilityDashboardSettings.__annotations__, dashboard_settings_data_tuple)))

    def update_profitability_dashboard_settings(self, dashboard_settings_id: int, dashboard_settings: ProfitabilityDashboardSettings) -> ProfitabilityDashboardSettings:
        query = text(f"""
            UPDATE {self.table_name}
            SET board_id = :board_id, name = :name, financial_year_start = :financial_year_start, financial_year_end = :financial_year_end, 
                forecast_period = :forecast_period, output_response = :output_response, publish_to_cfo = :publish_to_cfo, updated_at = CURRENT_TIMESTAMP
            WHERE id = :dashboard_settings_id
            RETURNING id, board_id, name, financial_year_start, financial_year_end, forecast_period, output_response, publish_to_cfo, created_at, updated_at;
        """)
        values = dashboard_settings.dict()
        values["dashboard_settings_id"] = dashboard_settings_id
        dashboard_settings_data_tuple = self.execute_query(query, values)
        return ProfitabilityDashboardSettings(**dict(zip(ProfitabilityDashboardSettings.__annotations__, dashboard_settings_data_tuple)))

    def delete_profitability_dashboard_settings(self, dashboard_settings_id: int) -> ProfitabilityDashboardSettings:
        query = text(f"""
            DELETE FROM {self.table_name} WHERE id = :dashboard_settings_id
            RETURNING id, board_id, name, financial_year_start, financial_year_end, forecast_period, output_response, publish_to_cfo, created_at, updated_at;
        """)
        values = {"dashboard_settings_id": dashboard_settings_id}
        dashboard_settings_data_tuple = self.execute_query(query, values)
        return ProfitabilityDashboardSettings(**dict(zip(ProfitabilityDashboardSettings.__annotations__, dashboard_settings_data_tuple)))

    def get_profitability_dashboard_settings_by_board_id(self, board_id: int) -> List[ProfitabilityDashboardSettings]:
        query = text(f"""
            SELECT * FROM {self.table_name} WHERE board_id = :board_id;
        """)
        values = {"board_id": board_id}
        dashboard_settings_data_list = self.execute_query_all(query, values)
        if dashboard_settings_data_list is None:
            return {}
        dashboard_settings_list = [ProfitabilityDashboardSettings(**dict(zip(ProfitabilityDashboardSettings.__annotations__, dashboard_settings_data))) for dashboard_settings_data in dashboard_settings_data_list]
        return dashboard_settings_list[0]
