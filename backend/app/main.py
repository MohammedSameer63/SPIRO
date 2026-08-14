from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    EmailAlreadyRegisteredError,
    WardNotFoundError,
    InvalidCredentialsError,
    AccountNotActiveError,
    HouseholdNotFoundError,
    HouseholdAccessDeniedError,
)

from app.api.routes.waste_category import (
    router as waste_category_router,
)
from app.api.routes.household import router as household_router
from app.api.routes.ward import router as ward_router
from app.api.routes.auth import router as auth_router
from app.core.config import settings

app = FastAPI(
    title=settings.project_name,
    version=settings.api_version,
)

@app.exception_handler(WardNotFoundError)
async def ward_not_found_handler(
    request: Request,
    exc: WardNotFoundError,
):
    return JSONResponse(
        status_code=404,
        content={
            "detail": str(exc),
        },
    )
    
@app.exception_handler(EmailAlreadyRegisteredError)
async def email_already_registered_handler(
    request: Request,
    exc: EmailAlreadyRegisteredError,
):
    return JSONResponse(
        status_code=409,
        content={
            "detail": str(exc),
        },
    )

@app.exception_handler(InvalidCredentialsError)
async def invalid_credentials_handler(
    request: Request,
    exc: InvalidCredentialsError,
):
    return JSONResponse(
        status_code=401,
        content={
            "detail": str(exc),
        },
    )

@app.exception_handler(AccountNotActiveError)
async def account_not_active_handler(
    request: Request,
    exc: AccountNotActiveError,
):
    return JSONResponse(
        status_code=403,
        content={
            "detail": str(exc),
        },
    )

@app.exception_handler(HouseholdNotFoundError)
async def household_not_found_handler(
    request: Request,
    exc: HouseholdNotFoundError,
):
    return JSONResponse(
        status_code=404,
        content={
            "detail": str(exc),
        },
    )


@app.exception_handler(HouseholdAccessDeniedError)
async def household_access_denied_handler(
    request: Request,
    exc: HouseholdAccessDeniedError,
):
    return JSONResponse(
        status_code=403,
        content={
            "detail": str(exc),
        },
    )

@app.get("/")
def home():
    return {
        "message": "SPIRO Backend API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


app.include_router(
    auth_router,
    prefix=settings.api_prefix,
)

app.include_router(
    ward_router,
    prefix="/api/v1",
)

app.include_router(
    household_router,
    prefix="/api/v1",
)

app.include_router(
    waste_category_router,
    prefix="/api/v1",
)