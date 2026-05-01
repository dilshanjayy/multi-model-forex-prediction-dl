import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from backend.db.database import SessionLocal
from backend.db import models, schemas
import math

def test():
    db = SessionLocal()
    try:
        trades = db.query(models.Trade).all()
        print(f"Found {len(trades)} trades")

        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl is not None and t.pnl > 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        total_pnl = sum((t.pnl if t.pnl is not None else 0.0) for t in trades)

        for t in trades:
            print(f"Trade {t.id}: timestamp={type(t.timestamp)} {t.timestamp}, pnl={t.pnl}, price={t.price}")
            if t.pnl is not None and (math.isnan(t.pnl) or math.isinf(t.pnl)):
                t.pnl = 0.0
            if math.isnan(t.price) or math.isinf(t.price):
                t.price = 0.0

        try:
            returned_dict = {
                "trades": trades,
                "total_trades": total_trades,
                "win_rate": round(float(win_rate), 2),
                "total_pnl": round(float(total_pnl), 2)
            }
            response = schemas.PortfolioResponse.model_validate(returned_dict)
            print("Serialization successful!")
            print(response.model_dump_json())
        except Exception as e:
            print("Serialization FAILED:")
            import traceback
            traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    test()
