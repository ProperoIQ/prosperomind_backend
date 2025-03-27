from typing import List
from sqlalchemy import text
from app.repositories.base_repository import BaseRepository
from app.models.forecast_response import ForecastResponse

class ForecastResponseRepository(BaseRepository):
    def __init__(self):
        super().__init__('ForecastResponse')
        create_table_query = text("""
            CREATE TABLE IF NOT EXISTS ForecastResponse (
                id SERIAL PRIMARY KEY,
                board_id INT REFERENCES Boards(id),
                name VARCHAR,
                first_level_filter VARCHAR,
                second_level_filter VARCHAR,
                financial_year_start VARCHAR,
                forecast_period INT,
                output_response TEXT,
                publish_to_cfo BOOLEAN,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            );
        """)
        self.create_table(create_table_query)

    def create_forecast_response(self, forecast_response: ForecastResponse) -> ForecastResponse:
        query = text("""
            INSERT INTO ForecastResponse (board_id, name, first_level_filter, second_level_filter,
                                        financial_year_start, forecast_period, output_response, publish_to_cfo,
                                        created_at, updated_at)
            VALUES (:board_id, :name, :first_level_filter, :second_level_filter, :financial_year_start, 
                    :forecast_period, :output_response, :publish_to_cfo, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING id, board_id, name, first_level_filter, second_level_filter,
                    financial_year_start, forecast_period, output_response, publish_to_cfo,
                    created_at, updated_at;
        """)

        # If forecast_name is not provided, use the name from forecast_response
        #if not forecast_response.name:
        # latest_forecast_response = self.get_latest_forecast_response()
        # latest_number = latest_forecast_response.id + 1 if latest_forecast_response else 1
        # automatic_name = f"Forecast setting {latest_number}"
        # forecast_response.name = automatic_name

        values = forecast_response.dict()
        forecast_response_data_tuple = self.execute_query(query, values)
        return ForecastResponse(**dict(zip(ForecastResponse.__annotations__, forecast_response_data_tuple)))


    def get_latest_forecast_response(self) -> ForecastResponse:
        query = text("""
            SELECT * FROM ForecastResponse ORDER BY id DESC LIMIT 1;
        """)
        forecast_response_data_tuple = self.execute_query(query)
        if forecast_response_data_tuple:
            return ForecastResponse(**dict(zip(ForecastResponse.__annotations__, forecast_response_data_tuple)))
        else:
            return None
        
    def get_forecast_response(self, forecast_response_id: int) -> ForecastResponse:
        query = text("""
            SELECT * FROM ForecastResponse WHERE id = :forecast_response_id;
        """)
        values = {"forecast_response_id": forecast_response_id}
        forecast_response_data_tuple = self.execute_query(query, values)
        return ForecastResponse(**dict(zip(ForecastResponse.__annotations__, forecast_response_data_tuple)))

    def update_forecast_response(self, forecast_response_id: int, forecast_response: ForecastResponse) -> ForecastResponse:
        query = text("""
            UPDATE ForecastResponse
            SET board_id = :board_id, name = :name, first_level_filter = :first_level_filter, 
                second_level_filter = :second_level_filter, financial_year_start = :financial_year_start, 
                forecast_period = :forecast_period, output_response = :output_response, 
                publish_to_cfo = :publish_to_cfo, updated_at = CURRENT_TIMESTAMP
            WHERE id = :forecast_response_id
            RETURNING id, board_id, name, first_level_filter, second_level_filter,
                      financial_year_start, forecast_period, output_response, publish_to_cfo,
                      created_at, updated_at;
        """)

        values = forecast_response.dict()
        values["forecast_response_id"] = forecast_response_id
        forecast_response_data_tuple = self.execute_query(query, values)
        return ForecastResponse(**dict(zip(ForecastResponse.__annotations__, forecast_response_data_tuple)))

    def delete_forecast_response(self, forecast_response_id: int) -> ForecastResponse:
        query = text("""
            DELETE FROM ForecastResponse WHERE id = :forecast_response_id
            RETURNING id, board_id, name, first_level_filter, second_level_filter,
                      financial_year_start, forecast_period, output_response, publish_to_cfo,
                      created_at, updated_at;
        """)
        values = {"forecast_response_id": forecast_response_id}
        forecast_response_data_tuple = self.execute_query(query, values)
        return ForecastResponse(**dict(zip(ForecastResponse.__annotations__, forecast_response_data_tuple)))

    def get_all_forecast_responses(self) -> List[ForecastResponse]:
        query = text("""
            SELECT * FROM ForecastResponse ORDER BY updated_at DESC;
        """)
        forecast_responses_data_list = self.execute_query_all(query)
        forecast_responses_list = [ForecastResponse(**dict(zip(ForecastResponse.__annotations__, forecast_response_data))) for forecast_response_data in forecast_responses_data_list]
        return forecast_responses_list
    
    def get_forecast_response_by_board_id(self, board_id: int) -> ForecastResponse:
        query = text("""
            SELECT * FROM ForecastResponse WHERE board_id = :board_id and publish_to_cfo = TRUE ORDER BY updated_at DESC LIMIT 1;
        """)
        values = {"board_id": board_id}
        forecast_response_data = self.execute_query(query, values)
        if forecast_response_data:
            forecast_response = ForecastResponse(**dict(zip(ForecastResponse.__annotations__, forecast_response_data)))
            return forecast_response
        else:
            return None
        
    def get_forecast_response_by_board_id_consultant(self, board_id: int) -> ForecastResponse:
        query = text("""
            SELECT * FROM ForecastResponse WHERE board_id = :board_id ORDER BY updated_at DESC;
        """)
        values = {"board_id": board_id}
        forecast_responses_data_list = self.execute_query_all(query, values)
        if forecast_responses_data_list:
            forecast_responses_list = [ForecastResponse(**dict(zip(ForecastResponse.__annotations__, forecast_response_data))) for forecast_response_data in forecast_responses_data_list]
            return forecast_responses_list
        else:
            return None

    def update_publish_to_cfo(self, forecast_response_id: int, publish_to_cfo: bool) -> ForecastResponse:
        query = text("""
            UPDATE ForecastResponse
            SET publish_to_cfo = :publish_to_cfo, updated_at = CURRENT_TIMESTAMP
            WHERE id = :forecast_response_id
            RETURNING id, board_id, name, first_level_filter, second_level_filter,
                    financial_year_start, forecast_period, output_response, publish_to_cfo,
                    created_at, updated_at;
        """)
        values = {"forecast_response_id": forecast_response_id, "publish_to_cfo": publish_to_cfo}
        forecast_response_data_tuple = self.execute_query(query, values)
        return ForecastResponse(**dict(zip(ForecastResponse.__annotations__, forecast_response_data_tuple)))
    
    
    