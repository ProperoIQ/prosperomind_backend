import hashlib
from sqlalchemy.sql import text
from app.repositories.base_repository import BaseRepository
from app.models.forecast_chat_response import ForecastChatResponse
from typing import Optional

class ForecastChatResponseRepository(BaseRepository):
    def __init__(self):
        super().__init__('Forecast_chat_response')
        self._create_table()

    def _create_table(self):
        create_table_query = text(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id SERIAL PRIMARY KEY,
                board_id INT REFERENCES Boards(id),
                forecast_response_id INT REFERENCES ForecastResponse(id),
                input_text TEXT,
                first_level_filter TEXT,
                second_level_filter TEXT,
                forecast_period INT,
                hash_key TEXT,
                name TEXT,
                response_content JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
            );
            """
        )
        self.create_table(create_table_query)

    def generate_hash_key(self, contents: bytes, input_text: str) -> str:
        hash_object = hashlib.sha256(contents + input_text.encode())
        return hash_object.hexdigest()

    async def check_existing_response(self, hash_key: str) -> Optional[ForecastChatResponse]:
        query = text(f"SELECT * FROM {self.table_name} WHERE hash_key = :hash_key")
        values = {"hash_key": hash_key}
        response_data_tuple = self.execute_query(query, values)
        return ForecastChatResponse(**dict(zip(ForecastChatResponse.__annotations__, response_data_tuple))) if response_data_tuple else None

    def save_response_to_database(self, board_id: int, forecast_response_id: int, input_text: str,
                                        first_level_filter: str, second_level_filter: str, forecast_period: int,
                                        name: str, response_content: dict, hash_key):
        query = text(
            f"""
            INSERT INTO {self.table_name} (board_id, forecast_response_id, input_text, first_level_filter,
                                            second_level_filter, forecast_period, hash_key, name, response_content)
            VALUES (:board_id, :forecast_response_id, :input_text, :first_level_filter, :second_level_filter,
                    :forecast_period, :hash_key, :name, :response_content)
            RETURNING id, board_id, forecast_response_id, input_text, first_level_filter, second_level_filter,
                      forecast_period, hash_key, name, response_content, created_at, updated_at;
            """
        )

        values = {
            "board_id": board_id,
            "forecast_response_id": forecast_response_id,
            "input_text": input_text,
            "first_level_filter": first_level_filter,
            "second_level_filter": second_level_filter,
            "forecast_period": forecast_period,
            "hash_key": hash_key,
            "name": name,
            "response_content": response_content
        }

        response_data_tuple = self.execute_query(query, values)
        # return ForecastChatResponse(**dict(zip(ForecastChatResponse.__annotations__, response_data_tuple)))
    

    async def update_response_in_database(self, response_id: int, response_content: dict) -> Optional[ForecastChatResponse]:
        query = text(
            f"""
            UPDATE {self.table_name} SET response_content = :response_content, updated_at = CURRENT_TIMESTAMP
            WHERE id = :response_id
            RETURNING id, board_id, forecast_response_id, input_text, first_level_filter, second_level_filter,
                      forecast_period, hash_key, name, response_content, created_at, updated_at;
            """
        )

        values = {
            "response_id": response_id,
            "response_content": response_content
        }

        response_data_tuple = self.execute_query(query, values)
        return ForecastChatResponse(**dict(zip(ForecastChatResponse.__annotations__, response_data_tuple))) if response_data_tuple else None

    async def delete_response_from_database(self, response_id: int) -> Optional[ForecastChatResponse]:
        query = text(
            f"""
            DELETE FROM {self.table_name} WHERE id = :response_id
            RETURNING id, board_id, forecast_response_id, input_text, first_level_filter, second_level_filter,
                      forecast_period, hash_key, name, response_content, created_at, updated_at;
            """
        )

        values = {
            "response_id": response_id
        }

        response_data_tuple = self.execute_query(query, values)
        return ForecastChatResponse(**dict(zip(ForecastChatResponse.__annotations__, response_data_tuple))) if response_data_tuple else None

    async def get_responses_for_board(self, board_id: int) -> Optional[ForecastChatResponse]:
        query = text(f"SELECT * FROM {self.table_name} WHERE board_id = :board_id")
        values = {"board_id": board_id}
        response_data = self.execute_query_all(query, values)
        return [ForecastChatResponse(**dict(zip(ForecastChatResponse.__annotations__, response_tuple))) for response_tuple in response_data]

    async def get_forecast_chat_response(self, response_id: int) -> Optional[ForecastChatResponse]:
        query = text(f"SELECT * FROM {self.table_name} WHERE id = :response_id")
        values = {"response_id": response_id}
        response_data_tuple = self.execute_query(query, values)
        return ForecastChatResponse(**dict(zip(ForecastChatResponse.__annotations__, response_data_tuple))) if response_data_tuple else None

    async def get_responses_for_using_forecast_id(self, forecast_response_id: int) -> Optional[ForecastChatResponse]:
        query = text(f"SELECT * FROM {self.table_name} WHERE forecast_response_id = :forecast_response_id")
        values = {"forecast_response_id": forecast_response_id}
        response_data = self.execute_query_all(query, values)
        return [ForecastChatResponse(**dict(zip(ForecastChatResponse.__annotations__, response_tuple))) for response_tuple in response_data]
