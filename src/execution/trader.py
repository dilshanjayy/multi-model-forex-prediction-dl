# type: ignore
import MetaTrader5 as mt5


def execute_market_order(symbol: str, lot_size: float, direction: str, atr: float, multiplier: float):
    """
    Sends a market order to MT5 with ATR-based Stop Loss and Take Profit.
    Includes auto-detection for Filling Mode.
    """
    if not mt5.initialize():
        return {"status": "error", "message": "MT5 Initialization failed"}

    # 1. Check Terminal Permissions
    terminal_info = mt5.terminal_info()
    if not terminal_info.trade_allowed:
        return {"status": "error", "message": "MT5 Error: 'Algo Trading' button is disabled in the terminal toolbar."}

    # 2. Get current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"status": "error", "message": f"Could not get tick for {symbol}. Check symbol name."}

    price = tick.ask if direction == 'BUY' else tick.bid

    # 3. Calculate Barriers
    dist = atr * multiplier
    sl = price - dist if direction == 'BUY' else price + dist
    tp = price + dist if direction == 'BUY' else price - dist

    # 4. Auto-Detect Filling Mode (Fixes broker-specific crashes)
    symbol_info = mt5.symbol_info(symbol)
    filling_mode = mt5.ORDER_FILLING_IOC # Default
    # 1 corresponds to SYMBOL_FILLING_FOK, 2 corresponds to SYMBOL_FILLING_IOC
    if symbol_info.filling_mode & 1:
        filling_mode = mt5.ORDER_FILLING_FOK
    elif symbol_info.filling_mode & 2:
        filling_mode = mt5.ORDER_FILLING_IOC
    else:
        filling_mode = mt5.ORDER_FILLING_RETURN

    # 5. Prepare Request
    order_type = mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot_size),
        "type": order_type,
        "price": price,
        "sl": round(sl, 5),
        "tp": round(tp, 5),
        "magic": 20240419,
        "comment": "Dashboard ML Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode,
    }

    # 6. Send Order
    result = mt5.order_send(request)

    if result is None:
        return {"status": "error", "message": "MT5 returned None. Request might be malformed."}

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {"status": "error", "message": f"Order failed [{result.retcode}]: {result.comment}"}

    return {"status": "success", "deal": result.deal, "price": result.price}

    return {"status": "success", "deal": result.deal, "price": result.price}
