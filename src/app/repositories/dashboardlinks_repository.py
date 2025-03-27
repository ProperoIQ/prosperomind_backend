from datetime import datetime
from typing import Optional, List
from sqlalchemy import text
from app.database import get_database_connection
from app.models.dashboardlink import DashboardLink, DashboardLinkCreate
from app.repositories.base_repository import BaseRepository

class DashboardLinkRepository(BaseRepository):
    def __init__(self):
        super().__init__('DashboardLinks')
        create_table_query = text(f"""
                                CREATE TABLE IF NOT EXISTS {self.table_name} (
                                    id SERIAL PRIMARY KEY,
                                    board_id INT NOT NULL,
                                    link TEXT NOT NULL,
                                    created_at TIMESTAMP,
                                    updated_at TIMESTAMP, 
                                    is_deleted BOOLEAN DEFAULT FALSE
                                );
        """)
        self.create_table(create_table_query)

    def create_dashboard_link(self, dashboard_link_create: DashboardLinkCreate):
        query = text(f"""
            INSERT INTO {self.table_name} (board_id, link, created_at, updated_at, is_deleted)
            VALUES (:board_id, :link, :created_at, :updated_at, :is_deleted)
            RETURNING id, board_id, link, created_at, updated_at, is_deleted;
        """)

        values = {
            "board_id": dashboard_link_create.board_id,
            "link": dashboard_link_create.link,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "is_deleted": False
        }

        created_link_tuple = self.execute_query(query, values)
        created_link = DashboardLink(**dict(zip(DashboardLink.__annotations__, created_link_tuple)))
        return created_link

    def get_dashboard_links_by_board(self, board_id: int):
        query = text(f"""
            SELECT id, board_id, link, created_at, updated_at, is_deleted
            FROM {self.table_name}
            WHERE board_id = :board_id AND is_deleted = FALSE;
        """)

        values = {"board_id": board_id}

        links_data_list = self.execute_query_all(query, values)
       
        return links_data_list

    def get_dashboard_link_by_board_and_id(self, board_id: int, link_id: int) -> Optional[DashboardLink]:
        query = text(f"""
            SELECT id, board_id, link, created_at, updated_at, is_deleted
            FROM {self.table_name}
            WHERE board_id = :board_id AND id = :link_id AND is_deleted = FALSE;
        """)

        values = {"board_id": board_id, "link_id": link_id}

        link_data = self.execute_query(query, values)

        if link_data:
            return  link_data
        return None

    def get_dashboard_link(self, link_id: int) -> Optional[DashboardLink]:
        query = text(f"""
            SELECT id, board_id, link, created_at, updated_at, is_deleted
            FROM {self.table_name}
            WHERE id = :link_id AND is_deleted = FALSE;
        """)

        values = {"link_id": link_id}

        link_data = self.execute_query(query, values)

        if link_data:
            return DashboardLink(**dict(zip(DashboardLink.__annotations__, link_data)))
        return None

    def update_dashboard_link(self, link_id: int, dashboard_link: DashboardLinkCreate) -> Optional[DashboardLink]:
        query = text(f"""
            UPDATE {self.table_name}
            SET link = :link, updated_at = :updated_at
            WHERE id = :link_id AND is_deleted = FALSE
            RETURNING id, board_id, link, created_at, updated_at, is_deleted;
        """)

        values = {
            "link": dashboard_link.link,
            "updated_at": datetime.utcnow(),
            "link_id": link_id
        }

        updated_link_data = self.execute_query(query, values)

        if updated_link_data:
            return DashboardLink(**dict(zip(DashboardLink.__annotations__, updated_link_data)))
        return None

    def delete_dashboard_link(self, link_id: int) -> Optional[DashboardLink]:
        query = text(f"""
            UPDATE {self.table_name}
            SET is_deleted = TRUE, updated_at = :updated_at
            WHERE id = :link_id
            RETURNING id, board_id, link, created_at, updated_at, is_deleted;
        """)

        values = {
            "updated_at": datetime.utcnow(),
            "link_id": link_id
        }

        deleted_link_data = self.execute_query(query, values)

        if deleted_link_data:
            return DashboardLink(**dict(zip(DashboardLink.__annotations__, deleted_link_data)))
        return None
