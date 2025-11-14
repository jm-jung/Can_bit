"""Select between simulated and real trading clients."""
from src.trading.binance_client import trader as simulated_trader
from src.trading.binance_real_client import binance_real


class TradingRouter:
    def __init__(self) -> None:
        self.mode = "SIM"  # SIM or REAL

    def set_sim(self) -> None:
        self.mode = "SIM"

    def set_real(self) -> None:
        self.mode = "REAL"

    def get_client(self):
        if self.mode == "REAL":
            return binance_real
        return simulated_trader


trading_router = TradingRouter()
