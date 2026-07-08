import jwt
from fastapi import HTTPException, Security, Request, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from config.settings import settings
from typing import Optional

security = HTTPBearer(auto_error=False)

def _decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        
        token_type = payload.get("type")
        if token_type != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token type must be access token"
            )
            
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Subject ID missing from token claims"
            )
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token"
        )

def get_current_user_id(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
    token: Optional[str] = Query(None, description="JWT token (for SSE/EventSource connections)")
) -> str:
    """
    Extracts and validates the JWT from either:
      - Authorization: Bearer <token> header  (normal REST calls)
      - ?token=<token>  query param           (EventSource / SSE connections)
    """
    if credentials:
        return _decode_token(credentials.credentials)
    
    if token:
        return _decode_token(token)
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated"
    )
