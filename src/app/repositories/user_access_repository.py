# app/repositories/user_access_repository.py
from typing import Any
from sqlalchemy import text
from app.repositories.base_repository import BaseRepository
from app.models.user_access import UserAccess

class UserAccessRepository(BaseRepository):
    def __init__(self):
        super().__init__('UserAccess')
        create_table_query = text("""
            CREATE TABLE IF NOT EXISTS UserAccess (
                id SERIAL PRIMARY KEY,
                client_user_id INT REFERENCES ClientUser(id),
                main_boards_access TEXT,
                bcf_access TEXT,
                boards_access TEXT,
                prompts_access TEXT
            );
        """)
        self.create_table(create_table_query)

    def create_user_access(self, user_access: UserAccess) -> Any:
        query = text("""
            INSERT INTO UserAccess (client_user_id, main_boards_access, bcf_access, boards_access, prompts_access)
            VALUES (:client_user_id, :main_boards_access, :bcf_access, :boards_access, :prompts_access)
            RETURNING id, client_user_id, main_boards_access, bcf_access, boards_access, prompts_access;
        """)

        values = {
            "client_user_id": user_access.client_user_id,
            "main_boards_access": user_access.main_boards_access,
            "bcf_access": user_access.bcf_access,
            "boards_access": user_access.boards_access,
            "prompts_access": user_access.prompts_access,
        }

        user_access_data_tuple = self.execute_query(query, values)
        user_access_instance = UserAccess(**dict(zip(UserAccess.__annotations__, user_access_data_tuple)))
        return user_access_instance

    def get_all_user_access(self) -> Any:
        query = text("""
            SELECT * FROM UserAccess;
        """)

        user_access_data_list = self.execute_query_all(query)
        user_access_list = [UserAccess(**dict(zip(UserAccess.__annotations__, access_data))) for access_data in user_access_data_list]
        return user_access_list

    def get_user_access(self, access_id: int) -> Any:
        query = text("""
            SELECT * FROM UserAccess WHERE id = :access_id;
        """)

        values = {"access_id": access_id}

        user_access_data_tuple = self.execute_query(query, values)
        if user_access_data_tuple is None:
            return None
        user_access_instance = UserAccess(**dict(zip(UserAccess.__annotations__, user_access_data_tuple)))
        return user_access_instance

    def update_user_access(self, access_id: int, user_access: UserAccess) -> Any:
        query = text("""
            UPDATE UserAccess
            SET client_user_id = :client_user_id, main_boards_access = :main_boards_access,
                bcf_access = :bcf_access, boards_access = :boards_access, prompts_access = :prompts_access
            WHERE id = :access_id
            RETURNING id, client_user_id, main_boards_access, bcf_access, boards_access, prompts_access;
        """)

        values = {
            "client_user_id": user_access.client_user_id,
            "main_boards_access": user_access.main_boards_access,
            "bcf_access": user_access.bcf_access,
            "boards_access": user_access.boards_access,
            "prompts_access": user_access.prompts_access,
            "access_id": access_id
        }

        user_access_data_tuple = self.execute_query(query, values)
        if user_access_data_tuple is None:
            return user_access_data_tuple
        user_access_instance = UserAccess(**dict(zip(UserAccess.__annotations__, user_access_data_tuple)))
        return user_access_instance

    def delete_user_access(self, access_id: int) -> Any:
        query = text("""
            DELETE FROM UserAccess WHERE id = :access_id
            RETURNING id, client_user_id, main_boards_access, bcf_access, boards_access, prompts_access;
        """)

        values = {"access_id": access_id}

        user_access_data_tuple = self.execute_query(query, values)
        if user_access_data_tuple is None:
            return user_access_data_tuple
        user_access_instance = UserAccess(**dict(zip(UserAccess.__annotations__, user_access_data_tuple)))
        return user_access_instance
