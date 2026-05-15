import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add the project root to the python path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.db.database import engine, Base, SessionLocal
from backend.api.routes import router as api_router
from backend.api.auth import router as auth_router, get_password_hash
from backend.api.admin import router as admin_router
from backend.db import models

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Falcon Trading API", version="1.0.0")


def seed_admin_user():
    """Automatically creates a default admin user if the database is empty."""
    db = SessionLocal()
    try:
        admin_exists = db.query(models.User).filter(models.User.is_admin == True).first()
        if not admin_exists:
            print("--- Seeding Default Admin User ---")
            default_admin = models.User(
                username="admin",
                email="admin@example.com",
                hashed_password=get_password_hash("admin123"),
                is_admin=True,
            )
            db.add(default_admin)
            db.commit()
            print("SUCCESS: Default Admin created (User: admin, Pass: admin123)")
    except Exception as e:
        print(f"Error seeding admin user: {e}")
    finally:
        db.close()


@app.on_event("startup")
async def startup_event():
    """Pre-load ML models and seed database on startup."""
    seed_admin_user()
    try:
        from src.data.sentiment_processor import get_sentiment_engine

        print("--- Pre-loading Sentiment Engine (FinBERT) ---")
        get_sentiment_engine()
        print("--- Sentiment Engine Ready ---")

        from backend.api.routes import preload_all_models

        preload_all_models()

    except Exception as e:
        print(f"Warning: Could not pre-load engines: {e}")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(admin_router, prefix="/api/v1/admin", tags=["admin"])
app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
