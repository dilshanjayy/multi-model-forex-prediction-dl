import os
import shutil
import zipfile
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.db.database import get_db
from backend.db import models
from backend.api.auth import get_current_admin_user
from backend.api.routes import MODELS_DIR, preload_all_models

router = APIRouter()

@router.get("/stats")
def get_global_stats(db: Session = Depends(get_db), admin: models.User = Depends(get_current_admin_user)):
    """Returns global trading statistics across all users (Admin only)."""
    try:
        total_trades = db.query(func.count(models.Trade.id)).scalar() or 0
        total_users = db.query(func.count(models.User.id)).scalar() or 0
        
        # Calculate global PnL
        total_pnl = db.query(func.sum(models.Trade.pnl)).scalar() or 0.0
        
        # Calculate global win rate
        winning_trades = db.query(func.count(models.Trade.id)).filter(models.Trade.pnl > 0).scalar() or 0
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        # Most used model (ignore historical dummy labels)
        most_used_model = db.query(models.Trade.model_used, func.count(models.Trade.id).label('count')) \
            .filter(models.Trade.model_used != "Live Execution") \
            .filter(models.Trade.model_used != "Manual") \
            .group_by(models.Trade.model_used) \
            .order_by(func.count(models.Trade.id).desc()) \
            .first()
            
        top_model = most_used_model[0] if most_used_model else "N/A"

        return {
            "total_users": total_users,
            "total_trades": total_trades,
            "global_pnl": round(total_pnl, 2),
            "global_win_rate": round(win_rate, 2),
            "top_model": top_model
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload_model")
async def upload_model(file: UploadFile = File(...), admin: models.User = Depends(get_current_admin_user)):
    """Accepts a .zip file containing a new model and extracts it to deployed_models."""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files are allowed.")
    
    # Save the uploaded zip file temporarily
    temp_zip_path = f"temp_{file.filename}"
    try:
        with open(temp_zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Extract the zip file
        model_name = file.filename.replace('.zip', '')
        extract_path = os.path.join(MODELS_DIR, model_name)
        
        if os.path.exists(extract_path):
            raise HTTPException(status_code=400, detail=f"A model named {model_name} already exists.")
            
        os.makedirs(extract_path, exist_ok=True)
        
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
        return {"status": "success", "message": f"Model '{model_name}' uploaded successfully."}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {str(e)}")
    finally:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)


@router.post("/reload_models")
def reload_models(admin: models.User = Depends(get_current_admin_user)):
    """Triggers a hot-reload of the model cache (Admin only)."""
    try:
        preload_all_models()
        return {"status": "success", "message": "Model cache successfully hot-reloaded."}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to reload models: {str(e)}")
