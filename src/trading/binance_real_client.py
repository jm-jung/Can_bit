"""Safe Binance real-client wrapper (dry run by default)."""
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, Literal, Optional

import ccxt

# 환경변수에서 Binance 설정 읽기
# BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_SYMBOL, BINANCE_SANDBOX_MODE, BINANCE_LIVE_TRADING
OrderSide = Literal["BUY", "SELL"]


class BinanceRealClient:
    """실제 Binance 주문을 수행할 수 있는 클라이언트 (기본값: dry-run 모드)."""

    def __init__(self):
        # 환경변수에서 설정 읽기
        self.api_key = os.getenv("BINANCE_API_KEY", "")
        self.api_secret = os.getenv("BINANCE_API_SECRET", "")
        self.symbol = os.getenv("BINANCE_SYMBOL", "BTC/USDT")
        
        # Sandbox 모드 설정 (기본값: true)
        sandbox_flag = os.getenv("BINANCE_SANDBOX_MODE", "true").lower() == "true"
        
        # Live Trading 모드 설정 (기본값: false)
        live_flag = os.getenv("BINANCE_LIVE_TRADING", "false").lower() == "true"
        
        # ccxt Binance exchange 초기화
        self.exchange = ccxt.binance({
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
        })
        
        # Sandbox 모드 설정
        if sandbox_flag:
            self.exchange.set_sandbox_mode(True)
            logging.info(f"[REAL CLIENT] Sandbox mode enabled for {self.symbol}")
        else:
            logging.warning(f"[REAL CLIENT] Mainnet mode enabled for {self.symbol} - 실제 자금 사용 가능!")
        
        # Live mode 설정 (환경변수에서 초기값 로드)
        self.live_mode = live_flag
        if self.live_mode:
            logging.warning("[REAL CLIENT] Live trading mode is ENABLED - 실제 주문이 실행될 수 있습니다!")
        else:
            logging.info("[REAL CLIENT] Live trading mode is DISABLED (dry-run mode)")
        
        # 메모리 상의 포지션 저장
        self.position: Optional[Dict] = None

    def get_position(self) -> Optional[Dict]:
        """현재 메모리 상의 포지션 반환."""
        return self.position

    def get_live_mode(self) -> bool:
        """Live mode 상태 반환."""
        return self.live_mode

    def enable_live_mode(self) -> None:
        """Live trading 모드 활성화 (실제 주문 실행 가능)."""
        self.live_mode = True
        logging.warning("[REAL CLIENT] Live trading mode ENABLED - 실제 주문이 실행될 수 있습니다!")

    def disable_live_mode(self) -> None:
        """Live trading 모드 비활성화 (dry-run 모드)."""
        self.live_mode = False
        logging.info("[REAL CLIENT] Live trading mode DISABLED (dry-run mode)")

    def create_order(self, side: OrderSide, price: float, amount: float) -> Dict:
        """
        주문 생성 (진입).
        
        Args:
            side: "BUY" or "SELL"
            price: 주문 가격 (참고용, MARKET 주문에서는 실제 체결가가 다를 수 있음)
            amount: 주문 수량
            
        Returns:
            주문 정보 딕셔너리
        """
        side_lower = side.lower()
        now = datetime.utcnow().isoformat()
        
        # Live mode가 False인 경우: 실제 API 호출하지 않음
        if not self.live_mode:
            logging.warning(f"[REAL DRY RUN] {side} {amount} {self.symbol} @ ~{price}")
            self.position = {
                "id": str(uuid.uuid4()),
                "side": side,
                "price": price,
                "amount": amount,
                "entry_time": now,
            }
            return {
                "id": self.position["id"],
                "side": side,
                "price": price,
                "amount": amount,
                "entry_time": now,
                "executed": False,
                "mode": "dry-run",
            }
        
        # Live mode == True → 실제 Binance에 주문 전송
        try:
            logging.info(f"[REAL LIVE] Creating {side} order: {amount} {self.symbol} @ MARKET")
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="market",
                side=side_lower,
                amount=amount,
            )
            
            # 체결 가격 결정 (average > price > 입력 price 순서)
            filled_price = order.get("average") or order.get("price") or price
            filled_price = float(filled_price)
            
            # 포지션 저장
            self.position = {
                "id": order.get("id") or str(uuid.uuid4()),
                "side": side,
                "price": filled_price,
                "amount": amount,
                "entry_time": now,
            }
            
            logging.info(f"[REAL LIVE] Order executed: {self.position['id']} @ {filled_price}")
            
            return {
                "id": self.position["id"],
                "side": self.position["side"],
                "price": self.position["price"],
                "amount": self.position["amount"],
                "entry_time": now,
                "executed": True,
                "mode": "live",
            }
        except Exception as e:
            logging.error(f"[REAL ERROR] create_order failed: {e}")
            raise

    def close_position(self, exit_price: float) -> Optional[Dict]:
        """
        포지션 청산.
        
        Args:
            exit_price: 청산 가격 (참고용, MARKET 주문에서는 실제 체결가가 다를 수 있음)
            
        Returns:
            청산 결과 딕셔너리 (포지션이 없으면 None)
        """
        if not self.position:
            return None
        
        entry = self.position
        side = entry["side"]
        amount = entry["amount"]
        entry_price = entry["price"]
        
        # 청산 방향 결정 (BUY 포지션이면 SELL, SELL 포지션이면 BUY)
        close_side = "sell" if side == "BUY" else "buy"
        
        # Live mode에 따라 실제 주문 실행 여부 결정
        if not self.live_mode:
            logging.warning(f"[REAL DRY RUN CLOSE] {close_side.upper()} {amount} {self.symbol} @ ~{exit_price}")
            exit_exec_price = exit_price
            executed = False
        else:
            try:
                logging.info(f"[REAL LIVE] Closing position: {close_side.upper()} {amount} {self.symbol} @ MARKET")
                order = self.exchange.create_order(
                    symbol=self.symbol,
                    type="market",
                    side=close_side,
                    amount=amount,
                )
                
                # 체결 가격 결정
                exit_exec_price = order.get("average") or order.get("price") or exit_price
                exit_exec_price = float(exit_exec_price)
                executed = True
                
                logging.info(f"[REAL LIVE] Position closed: {order.get('id')} @ {exit_exec_price}")
            except Exception as e:
                logging.error(f"[REAL ERROR] close_position failed: {e}")
                raise
        
        # PnL 계산
        if side == "BUY":
            pnl = (exit_exec_price - entry_price) * amount
        else:  # SELL
            pnl = (entry_price - exit_exec_price) * amount
        
        # 포지션 리셋
        self.position = None
        
        return {
            "entry": entry,
            "exit_price": exit_exec_price,
            "exit_time": datetime.utcnow().isoformat(),
            "pnl": float(pnl),
            "executed": executed,
        }


binance_real = BinanceRealClient()

