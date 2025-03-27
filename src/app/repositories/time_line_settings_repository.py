# app/repositories/time_line_settings_repository.py
from typing import Any
from sqlalchemy import text
from app.repositories.base_repository import BaseRepository
from app.models.time_line_settings import TimeLineSettings

class TimeLineSettingsRepository(BaseRepository):
    def __init__(self):
        super().__init__('TimeLineSettings')
        create_table_query = text("""
            CREATE TABLE IF NOT EXISTS TimeLineSettings (
                id SERIAL PRIMARY KEY,
                board_id INT REFERENCES Boards(id),
                configuration_details TEXT,
                name VARCHAR(255),
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            );
        """)
        self.create_table(create_table_query)

    def create_time_line_settings(self, time_line_settings: TimeLineSettings) -> Any:
        query = text("""
            INSERT INTO TimeLineSettings (board_id, configuration_details, name, created_at, updated_at)
            VALUES (:board_id, :configuration_details, :name, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING id, board_id, configuration_details, name, created_at, updated_at;
        """)

        values = {
            "board_id": time_line_settings.board_id,
            "configuration_details": time_line_settings.configuration_details,
            "name": time_line_settings.name,
        }

        time_line_settings_data_tuple = self.execute_query(query, values)
        time_line_settings_instance = TimeLineSettings(**dict(zip(TimeLineSettings.__annotations__, time_line_settings_data_tuple)))
        return time_line_settings_instance

    def get_all_time_line_settings(self) -> Any:
        query = text("""
            SELECT * FROM TimeLineSettings;
        """)

        time_line_settings_data_list = self.execute_query_all(query)
        time_line_settings_list = []

        for settings_data in time_line_settings_data_list:
            time_line_settings_instance = TimeLineSettings(**dict(zip(TimeLineSettings.__annotations__, settings_data)))
            try:
                time_line_settings_instance.configuration_details = eval(time_line_settings_instance.configuration_details)
            except Exception as ex:
                time_line_settings_instance.configuration_details = time_line_settings_instance.configuration_details
            time_line_settings_list.append(time_line_settings_instance)

        return time_line_settings_list

    def get_time_line_settings(self, settings_id: int) -> Any:
        query = text("""
            SELECT * FROM TimeLineSettings WHERE id = :settings_id;
        """)

        values = {"settings_id": settings_id}

        time_line_settings_data_tuple = self.execute_query(query, values)
        if time_line_settings_data_tuple is None:
            return None
        time_line_settings_instance = TimeLineSettings(**dict(zip(TimeLineSettings.__annotations__, time_line_settings_data_tuple)))
        try:
            time_line_settings_instance.configuration_details = eval(time_line_settings_instance.configuration_details)
        except Exception as ex:
            return time_line_settings_instance
        return time_line_settings_instance

    def update_time_line_settings(self, settings_id: int, time_line_settings: TimeLineSettings) -> Any:
        query = text("""
            UPDATE TimeLineSettings
            SET board_id = :board_id, configuration_details = :configuration_details, name = :name, updated_at = CURRENT_TIMESTAMP
            WHERE id = :settings_id
            RETURNING id, board_id, configuration_details, name, created_at, updated_at;
        """)

        values = {
            "board_id": time_line_settings.board_id,
            "configuration_details": time_line_settings.configuration_details,
            "name": time_line_settings.name,
            "settings_id": settings_id
        }

        time_line_settings_data_tuple = self.execute_query(query, values)
        if time_line_settings_data_tuple is None:
            return time_line_settings_data_tuple
        time_line_settings_instance = TimeLineSettings(**dict(zip(TimeLineSettings.__annotations__, time_line_settings_data_tuple)))
        return time_line_settings_instance

    def delete_time_line_settings(self, settings_id: int) -> Any:
        query = text("""
            DELETE FROM TimeLineSettings WHERE id = :settings_id
            RETURNING id, board_id, configuration_details, name, created_at, updated_at;
        """)

        values = {"settings_id": settings_id}

        time_line_settings_data_tuple = self.execute_query(query, values)
        if time_line_settings_data_tuple is None:
            return time_line_settings_data_tuple
        time_line_settings_instance = TimeLineSettings(**dict(zip(TimeLineSettings.__annotations__, time_line_settings_data_tuple)))
        return time_line_settings_instance
