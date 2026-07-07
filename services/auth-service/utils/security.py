import jwt
from datetime import datetime, timedelta
from typing import Union, Dict, Any
from passlib.context import CryptContext
from config.settings import settings

# Initialize password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_jwt_token(data: dict, secret: str, expires_delta: int) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(seconds=expires_delta)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, secret, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

def create_access_token(user_id: str, email: str) -> str:
    return create_jwt_token(
        data={"sub": user_id, "email": email, "type": "access"},
        secret=settings.JWT_SECRET,
        expires_delta=settings.JWT_ACCESS_EXPIRE
    )

def create_refresh_token(user_id: str) -> str:
    return create_jwt_token(
        data={"sub": user_id, "type": "refresh"},
        secret=settings.JWT_REFRESH_SECRET,
        expires_delta=settings.JWT_REFRESH_EXPIRE
    )

def decode_token(token: str, secret: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, secret, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise Exception("Token has expired")
    except jwt.InvalidTokenError:
        raise Exception("Invalid token")
