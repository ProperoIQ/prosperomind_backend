# app/dependencies.py

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from app.database import get_database_connection

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_db():
    db = get_database_connection()
    try:
        yield db
    finally:
        db.close()
