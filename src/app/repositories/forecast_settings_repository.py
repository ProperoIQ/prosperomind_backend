from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel
from sqlalchemy import text
from app.repositories.base_repository import BaseRepository
from app.models.forecast_settings import ForecastSettings

class ForecastSettingsRepository(BaseRepository):
    def __init__(self):
        super().__init__('ForecastSettings')
        create_table_query = text("""
            CREATE TABLE IF NOT EXISTS ForecastSettings (
                id SERIAL PRIMARY KEY,
                board_id INT REFERENCES Boards(id),
                financial_year_start VARCHAR,
                financial_year_end VARCHAR,
                date_variable VARCHAR,
                independent_variables VARCHAR[],
                dependent_variable VARCHAR,
                forecast_length INT,
                budget INT,
                model_level_1 VARCHAR,
                model_level_2 VARCHAR,
                prediction_interval FLOAT,
                tuning_parameters JSONB,
                preprocessing_options JSONB,
                automatic_forecasting BOOLEAN,
                evaluation_metrics VARCHAR[],
                forecast_horizon INT,
                visualization_options JSONB,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            );
        """)
        self.create_table(create_table_query)

    def create_forecast_settings(self, forecast_settings: ForecastSettings) -> ForecastSettings:
        query = text("""
            INSERT INTO ForecastSettings (board_id, financial_year_start, financial_year_end, 
                                          date_variable, independent_variables, dependent_variable, forecast_length, 
                                          budget,
                                          model_level_1, model_level_2, prediction_interval, 
                                          tuning_parameters, preprocessing_options, automatic_forecasting, 
                                          evaluation_metrics, forecast_horizon, visualization_options, 
                                          created_at, updated_at)
            VALUES (:board_id, :financial_year_start, :financial_year_end, 
                    :date_variable, :independent_variables, :dependent_variable, :forecast_length, 
                    :budget,
                    :model_level_1, :model_level_2, :prediction_interval, 
                    :tuning_parameters, :preprocessing_options, :automatic_forecasting, 
                    :evaluation_metrics, :forecast_horizon, :visualization_options, 
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING id, board_id, financial_year_start, financial_year_end, 
                      date_variable, independent_variables, dependent_variable, forecast_length, 
                      budget,
                      model_level_1, model_level_2, prediction_interval, 
                      tuning_parameters, preprocessing_options, automatic_forecasting, 
                      evaluation_metrics, forecast_horizon, visualization_options, 
                      created_at, updated_at;
        """)

        values = forecast_settings.dict()
        forecast_settings_data_tuple = self.execute_query(query, values)
        return ForecastSettings(**dict(zip(ForecastSettings.__annotations__, forecast_settings_data_tuple)))

    def get_forecast_settings(self, forecast_settings_id: int) -> ForecastSettings:
        query = text("""
            SELECT * FROM ForecastSettings WHERE id = :forecast_settings_id;
        """)
        values = {"forecast_settings_id": forecast_settings_id}
        forecast_settings_data_tuple = self.execute_query(query, values)
        if forecast_settings_data_tuple is None:
            return {}
        return ForecastSettings(**dict(zip(ForecastSettings.__annotations__, forecast_settings_data_tuple)))

    def update_forecast_settings(self, forecast_settings_id: int, forecast_settings: ForecastSettings) -> ForecastSettings:
        query = text("""
            UPDATE ForecastSettings
            SET board_id = :board_id, financial_year_start = :financial_year_start, financial_year_end = :financial_year_end, 
                date_variable = :date_variable, independent_variables = :independent_variables, dependent_variable = :dependent_variable, forecast_length = :forecast_length, 
                budget = :budget,
                model_level_1 = :model_level_1, model_level_2 = :model_level_2, prediction_interval = :prediction_interval, 
                tuning_parameters = :tuning_parameters, preprocessing_options = :preprocessing_options, automatic_forecasting = :automatic_forecasting, 
                evaluation_metrics = :evaluation_metrics, forecast_horizon = :forecast_horizon, visualization_options = :visualization_options, 
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :forecast_settings_id
            RETURNING id, board_id, financial_year_start, financial_year_end, 
                      date_variable, independent_variables, dependent_variable, forecast_length, 
                      budget,
                      model_level_1, model_level_2, prediction_interval, 
                      tuning_parameters, preprocessing_options, automatic_forecasting, 
                      evaluation_metrics, forecast_horizon, visualization_options, 
                      created_at, updated_at;
        """)

        values = forecast_settings.dict()
        values["forecast_settings_id"] = forecast_settings_id
        forecast_settings_data_tuple = self.execute_query(query, values)
        return ForecastSettings(**dict(zip(ForecastSettings.__annotations__, forecast_settings_data_tuple)))

    def delete_forecast_settings(self, forecast_settings_id: int) -> ForecastSettings:
        query = text("""
            DELETE FROM ForecastSettings WHERE id = :forecast_settings_id
            RETURNING id, board_id, financial_year_start, financial_year_end, 
                      date_variable, independent_variables, dependent_variable, forecast_length, 
                      budget,
                      model_level_1, model_level_2, prediction_interval, 
                      tuning_parameters, preprocessing_options, automatic_forecasting, 
                      evaluation_metrics, forecast_horizon, visualization_options, 
                      created_at, updated_at;
        """)
        values = {"forecast_settings_id": forecast_settings_id}
        forecast_settings_data_tuple = self.execute_query(query, values)
        return ForecastSettings(**dict(zip(ForecastSettings.__annotations__, forecast_settings_data_tuple)))

    def get_forecast_settings_by_board_id(self, board_id: int) -> List[ForecastSettings]:
        query = text("""
            SELECT * FROM ForecastSettings WHERE board_id = :board_id;
        """)
        values = {"board_id": board_id}
        forecast_settings_data_list = self.execute_query_all(query, values)
        if forecast_settings_data_list is None:
            return {}
        forecast_settings_list = [ForecastSettings(**dict(zip(ForecastSettings.__annotations__, forecast_settings_data))) for forecast_settings_data in forecast_settings_data_list]
        return forecast_settings_list[0]
