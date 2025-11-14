from datetime import datetime
from pydantic import BaseModel


class OHLCVCandle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

