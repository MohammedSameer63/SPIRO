from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.exceptions import (
    EmailAlreadyRegisteredError,
    WardNotFoundError,
    InvalidCredentialsError,
    AccountNotActiveError,
    HouseholdNotFoundError,
    HouseholdAccessDeniedError,
    InvalidReportImageError,
    ReportNotFoundError,
    WasteCategoryNotFoundError,
    PredictionAlreadyExistsError,
    InvalidCollectionScheduleError,
    AssignmentAlreadyExistsError,
    AssignmentNotFoundError,
    InvalidWorkerError,
    WorkerNotFoundError,
    WorkerWardAlreadyExistsError,
    WorkerWardAssignmentNotFoundError,
)

from app.api.routes.waste_category import (
    router as waste_category_router,
)
from app.api.routes.waste_report import (
    router as waste_report_router,
)
from app.api.routes.internal_prediction import (
    router as internal_prediction_router,
)
from app.api.routes.collection_schedule import (
    router as collection_schedule_router,
)
from app.api.routes.worker_ward import (
    router as worker_ward_router,
)
from app.api.routes.assignment import router as assignment_router
from app.api.routes.household import router as household_router
from app.api.routes.ward import router as ward_router
from app.api.routes.auth import router as auth_router
from app.core.config import settings

app = FastAPI(
    title=settings.project_name,
    version=settings.api_version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

@app.exception_handler(InvalidReportImageError)
async def invalid_report_image_handler(
    request: Request,
    exc: InvalidReportImageError,
):
    return JSONResponse(
        status_code=400,
        content={
            "detail": str(exc),
        },
    )

@app.exception_handler(ReportNotFoundError)
async def report_not_found_handler(
    request: Request,
    exc: ReportNotFoundError,
):
    return JSONResponse(
        status_code=404,
        content={
            "detail": str(exc),
        },
    )


@app.exception_handler(WasteCategoryNotFoundError)
async def waste_category_not_found_handler(
    request: Request,
    exc: WasteCategoryNotFoundError,
):
    return JSONResponse(
        status_code=404,
        content={
            "detail": str(exc),
        },
    )


@app.exception_handler(PredictionAlreadyExistsError)
async def prediction_already_exists_handler(
    request: Request,
    exc: PredictionAlreadyExistsError,
):
    return JSONResponse(
        status_code=409,
        content={
            "detail": str(exc),
        },
    )

@app.exception_handler(InvalidCollectionScheduleError)
async def invalid_collection_schedule_handler(
    request: Request,
    exc: InvalidCollectionScheduleError,
):
    return JSONResponse(
        status_code=400,
        content={
            "detail": str(exc),
        },
    )

@app.exception_handler(AssignmentNotFoundError)
async def assignment_not_found_handler(
    request: Request,
    exc: AssignmentNotFoundError,
):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)},
    )


@app.exception_handler(AssignmentAlreadyExistsError)
async def assignment_already_exists_handler(
    request: Request,
    exc: AssignmentAlreadyExistsError,
):
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": str(exc)},
    )


@app.exception_handler(InvalidWorkerError)
async def invalid_worker_handler(
    request: Request,
    exc: InvalidWorkerError,
):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )

@app.exception_handler(WardNotFoundError)
async def ward_not_found_handler(
    request,
    exc,
):
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc)},
    )

@app.exception_handler(WorkerWardAlreadyExistsError)
async def worker_ward_exists_handler(
    request,
    exc,
):
    return JSONResponse(
        status_code=409,
        content={"detail": str(exc)},
    )


@app.exception_handler(
    WorkerWardAssignmentNotFoundError
)
async def worker_ward_assignment_not_found_handler(
    request,
    exc,
):
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc)},
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

app.include_router(
    waste_report_router,
    prefix="/api/v1",
)

app.include_router(
    internal_prediction_router,
    prefix=settings.api_prefix,
)

app.include_router(
    collection_schedule_router,
    prefix=settings.api_prefix,
)

app.include_router(
    assignment_router,
    prefix=settings.api_prefix,
)

app.include_router(worker_ward_router)