import hashlib
import logging
from datetime import datetime, timedelta
from fastapi import HTTPException, status
from config.db import get_db
from models.user import UserRegister, UserLogin, UserOut, TokenResponse
from utils.security import hash_password, verify_password, create_access_token, create_refresh_token, decode_token
from config.settings import settings

logger = logging.getLogger(__name__)

def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()

class AuthController:
    @staticmethod
    async def register(user_data: UserRegister) -> UserOut:
        db = await get_db()
        
        # Check if user already exists
        res = await db.table("users").select("id").eq("email", user_data.email).execute()
        if res.data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Hash the password
        pwd_hash = hash_password(user_data.password)

        # Insert user
        insert_res = await db.table("users").insert({
            "email": user_data.email,
            "password_hash": pwd_hash,
            "first_name": user_data.first_name,
            "last_name": user_data.last_name
        }).execute()
        
        if not insert_res.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to register user"
            )
            
        row = insert_res.data[0]
        # Parse created_at datetime
        created_at_dt = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
        
        return UserOut(
            id=row["id"],
            email=row["email"],
            first_name=row.get("first_name"),
            last_name=row.get("last_name"),
            created_at=created_at_dt
        )

    @staticmethod
    async def login(credentials: UserLogin) -> TokenResponse:
        db = await get_db()
        
        # Fetch user
        res = await db.table("users").select("id, email, password_hash, first_name, last_name, created_at").eq("email", credentials.email).execute()
        if not res.data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
            
        row = res.data[0]
        if not verify_password(credentials.password, row["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )

        user_id = str(row["id"])
        email = row["email"]
        created_at_dt = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
        
        user_out = UserOut(
            id=row["id"],
            email=row["email"],
            first_name=row.get("first_name"),
            last_name=row.get("last_name"),
            created_at=created_at_dt
        )

        # Generate tokens
        access_token = create_access_token(user_id, email)
        refresh_token = create_refresh_token(user_id)

        # Store refresh token hash
        rt_hash = _hash_token(refresh_token)
        expires_at = (datetime.utcnow() + timedelta(seconds=settings.JWT_REFRESH_EXPIRE)).isoformat()

        await db.table("refresh_tokens").insert({
            "user_id": user_id,
            "token_hash": rt_hash,
            "expires_at": expires_at
        }).execute()

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user=user_out
        )

    @staticmethod
    async def refresh(refresh_token: str) -> TokenResponse:
        try:
            payload = decode_token(refresh_token, settings.JWT_REFRESH_SECRET)
            user_id = payload.get("sub")
            if not user_id:
                raise Exception("Missing subject field")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid refresh token: {e}"
            )

        rt_hash = _hash_token(refresh_token)
        db = await get_db()

        # Check if token exists, is not revoked and not expired
        res = await db.table("refresh_tokens").select("id, revoked, expires_at").eq("token_hash", rt_hash).eq("user_id", user_id).execute()
        if not res.data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token not found"
            )
            
        row = res.data[0]
        token_id = row["id"]
        revoked = row["revoked"]
        expires_at_str = row["expires_at"]
        
        if revoked:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has been revoked"
            )
            
        # Check expiry
        expires_at_dt = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
        # Strip timezone for utc comparison
        if expires_at_dt.tzinfo:
            expires_at_dt = expires_at_dt.replace(tzinfo=None)
            
        if expires_at_dt < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has expired"
            )

        # Fetch user
        user_res = await db.table("users").select("id, email, first_name, last_name, created_at").eq("id", user_id).execute()
        if not user_res.data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
            
        user_row = user_res.data[0]
        user_created_dt = datetime.fromisoformat(user_row["created_at"].replace("Z", "+00:00"))
        
        user_out = UserOut(
            id=user_row["id"],
            email=user_row["email"],
            first_name=user_row.get("first_name"),
            last_name=user_row.get("last_name"),
            created_at=user_created_dt
        )

        # Revoke current token
        await db.table("refresh_tokens").update({"revoked": True}).eq("id", token_id).execute()

        # Generate new tokens
        new_access_token = create_access_token(user_id, user_out.email)
        new_refresh_token = create_refresh_token(user_id)

        # Store new refresh token hash
        new_rt_hash = _hash_token(new_refresh_token)
        new_expires_at = (datetime.utcnow() + timedelta(seconds=settings.JWT_REFRESH_EXPIRE)).isoformat()

        await db.table("refresh_tokens").insert({
            "user_id": user_id,
            "token_hash": new_rt_hash,
            "expires_at": new_expires_at
        }).execute()

        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            user=user_out
        )

    @staticmethod
    async def logout(refresh_token: str) -> dict:
        rt_hash = _hash_token(refresh_token)
        db = await get_db()
        await db.table("refresh_tokens").update({"revoked": True}).eq("token_hash", rt_hash).execute()
        return {"status": "success", "message": "Logged out successfully"}
