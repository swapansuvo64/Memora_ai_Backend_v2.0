from fastapi import APIRouter
from models.user import UserRegister, UserLogin, UserOut, TokenResponse, RefreshRequest
from controllers.auth_controller import AuthController

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

@router.post("/register", response_model=UserOut, status_code=201)
async def register(user_data: UserRegister):
    return await AuthController.register(user_data)

@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    return await AuthController.login(credentials)

@router.post("/refresh", response_model=TokenResponse)
async def refresh(refresh_data: RefreshRequest):
    return await AuthController.refresh(refresh_data.refresh_token)

@router.post("/logout")
async def logout(refresh_data: RefreshRequest):
    return await AuthController.logout(refresh_data.refresh_token)
