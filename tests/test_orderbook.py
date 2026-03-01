"""
tests/test_orderbook.py

Proves correctness of the order book and matching engine from first principles.
Each test asserts exact outcomes for known input sequences.
Price-time priority must be provably correct.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import pytest
from engine.core.order import Order, Fill, Side, OrderType
from engine.core.order_book import OrderBook
from engine.core.matching_engine import MatchingEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_limit(agent_id, side, price, qty) -> Order:
    return Order(
        agent_id=agent_id, side=side,
        order_type=OrderType.LIMIT,
        price=price, qty=qty
    )

def make_market(agent_id, side, qty) -> Order:
    return Order(
        agent_id=agent_id, side=side,
        order_type=OrderType.MARKET,
        price=0.0, qty=qty
    )

def make_cancel(agent_id, target_id) -> Order:
    return Order(
        agent_id=agent_id, side=Side.BID,
        order_type=OrderType.CANCEL,
        price=0.0, qty=1,
        cancel_target=target_id,
    )

def fresh_engine():
    book = OrderBook()
    fills = []
    engine = MatchingEngine(book, on_fill=fills.append)
    return book, engine, fills


# ---------------------------------------------------------------------------
# Order book internal state
# ---------------------------------------------------------------------------

class TestOrderBookBasics:

    def test_best_bid_is_highest(self):
        book = OrderBook()
        book.add(make_limit("a", Side.BID, 100.0, 10))
        book.add(make_limit("b", Side.BID, 101.0, 5))
        book.add(make_limit("c", Side.BID, 99.0,  8))
        assert book.best_bid.price == 101.0

    def test_best_ask_is_lowest(self):
        book = OrderBook()
        book.add(make_limit("a", Side.ASK, 105.0, 10))
        book.add(make_limit("b", Side.ASK, 103.0, 5))
        book.add(make_limit("c", Side.ASK, 108.0, 8))
        assert book.best_ask.price == 103.0

    def test_mid_price_and_spread(self):
        book = OrderBook()
        book.add(make_limit("a", Side.BID, 100.0, 10))
        book.add(make_limit("b", Side.ASK, 102.0, 10))
        assert book.mid_price == 101.0
        assert book.spread == 2.0

    def test_empty_book_returns_none(self):
        book = OrderBook()
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.mid_price is None
        assert book.spread is None

    def test_cancel_removes_order(self):
        book = OrderBook()
        o = make_limit("a", Side.BID, 100.0, 10)
        book.add(o)
        assert book.best_bid is not None
        book.cancel(o.order_id)
        assert book.best_bid is None

    def test_cancel_nonexistent_is_safe(self):
        book = OrderBook()
        result = book.cancel("nonexistent-id")
        assert result is None

    def test_depth_snapshot_aggregates_levels(self):
        book = OrderBook()
        book.add(make_limit("a", Side.BID, 100.0, 10))
        book.add(make_limit("b", Side.BID, 100.0, 5))  # same price level
        book.add(make_limit("c", Side.BID, 99.0,  8))
        book.add(make_limit("d", Side.ASK, 101.0, 3))
        snap = book.depth_snapshot(n_levels=10)
        # Two bids at 100 should be aggregated
        bid_prices = [p for p, _ in snap["bids"]]
        assert bid_prices == sorted(bid_prices, reverse=True)
        assert 100.0 in bid_prices
        assert snap["bids"][0][1] == 15  # 10 + 5 at 100

    def test_len_tracks_live_orders(self):
        book = OrderBook()
        o1 = make_limit("a", Side.BID, 100.0, 5)
        o2 = make_limit("b", Side.ASK, 105.0, 5)
        book.add(o1)
        book.add(o2)
        assert len(book) == 2
        book.cancel(o1.order_id)
        assert len(book) == 1


# ---------------------------------------------------------------------------
# Price-time priority
# ---------------------------------------------------------------------------

class TestPriceTimePriority:

    def test_same_price_fifo(self):
        """Two bids at same price: earlier one fills first."""
        book, engine, fills = fresh_engine()
        b1 = make_limit("buyer1", Side.BID, 100.0, 5)
        b2 = make_limit("buyer2", Side.BID, 100.0, 5)
        # b1 arrives first
        engine.process([b1])
        engine.process([b2])

        # Sell 5 — should fill against b1 (earlier timestamp)
        engine.process([make_market("seller", Side.ASK, 5)])

        assert len(fills) == 1
        assert fills[0].buy_agent_id == "buyer1"

    def test_better_price_fills_first(self):
        """Higher bid fills before lower bid."""
        book, engine, fills = fresh_engine()
        b_low  = make_limit("low",  Side.BID, 99.0,  10)
        b_high = make_limit("high", Side.BID, 101.0, 10)
        engine.process([b_low, b_high])

        engine.process([make_market("seller", Side.ASK, 5)])
        assert fills[-1].buy_agent_id == "high"


# ---------------------------------------------------------------------------
# Limit order matching
# ---------------------------------------------------------------------------

class TestLimitMatching:

    def test_crossing_limit_fills_immediately(self):
        """Incoming bid above resting ask → crosses immediately."""
        book, engine, fills = fresh_engine()
        engine.process([make_limit("seller", Side.ASK,  99.0, 10)])
        engine.process([make_limit("buyer",  Side.BID, 100.0, 10)])

        assert len(fills) == 1
        assert fills[0].price == 99.0   # resting price
        assert fills[0].qty   == 10

    def test_partial_fill_rests_remainder(self):
        """Bid for 10, only 6 available on ask side → 6 filled, 4 rests."""
        book, engine, fills = fresh_engine()
        engine.process([make_limit("seller", Side.ASK, 100.0, 6)])
        engine.process([make_limit("buyer",  Side.BID, 101.0, 10)])

        assert len(fills) == 1
        assert fills[0].qty == 6
        # Remaining 4 should be on the bid side
        assert book.best_bid is not None
        assert book.best_bid.remaining_qty == 4
        assert book.best_bid.agent_id == "buyer"

    def test_no_cross_below_price(self):
        """Bid at 98 does not cross ask at 100 — both rest."""
        book, engine, fills = fresh_engine()
        engine.process([make_limit("seller", Side.ASK, 100.0, 10)])
        engine.process([make_limit("buyer",  Side.BID,  98.0, 10)])

        assert len(fills) == 0
        assert book.best_bid.price == 98.0
        assert book.best_ask.price == 100.0

    def test_multiple_levels_consumed(self):
        """Market buy for 25 sweeps through 3 ask levels."""
        book, engine, fills = fresh_engine()
        engine.process([
            make_limit("s1", Side.ASK, 100.0, 10),
            make_limit("s2", Side.ASK, 101.0, 10),
            make_limit("s3", Side.ASK, 102.0, 10),
        ])
        engine.process([make_market("buyer", Side.BID, 25)])

        assert sum(f.qty for f in fills) == 25
        prices = [f.price for f in fills]
        assert prices == [100.0, 101.0, 102.0]

    def test_self_trade_prevention_by_design(self):
        """
        The LOB does NOT prevent self-trades — that is an exchange policy,
        not a matching engine responsibility. This test just documents behavior.
        """
        book, engine, fills = fresh_engine()
        engine.process([make_limit("agent1", Side.ASK, 100.0, 5)])
        engine.process([make_limit("agent1", Side.BID, 101.0, 5)])
        # A fill does happen here — same agent on both sides
        assert len(fills) == 1


# ---------------------------------------------------------------------------
# Market orders
# ---------------------------------------------------------------------------

class TestMarketOrders:

    def test_market_buy_takes_best_ask(self):
        book, engine, fills = fresh_engine()
        engine.process([make_limit("seller", Side.ASK, 100.0, 10)])
        engine.process([make_market("buyer",  Side.BID, 5)])

        assert len(fills) == 1
        assert fills[0].price == 100.0
        assert fills[0].qty   == 5
        assert book.best_ask.remaining_qty == 5  # 10 - 5

    def test_market_sell_takes_best_bid(self):
        book, engine, fills = fresh_engine()
        engine.process([make_limit("buyer",  Side.BID, 100.0, 10)])
        engine.process([make_market("seller", Side.ASK, 3)])

        assert len(fills) == 1
        assert fills[0].price == 100.0
        assert fills[0].qty   == 3

    def test_market_order_on_empty_book_does_nothing(self):
        book, engine, fills = fresh_engine()
        engine.process([make_market("buyer", Side.BID, 5)])
        assert len(fills) == 0

    def test_market_partial_fill_when_insufficient_liquidity(self):
        """Market order for 20 with only 7 on book — fills 7, no error."""
        book, engine, fills = fresh_engine()
        engine.process([make_limit("seller", Side.ASK, 100.0, 7)])
        engine.process([make_market("buyer",  Side.BID, 20)])

        assert len(fills) == 1
        assert fills[0].qty == 7


# ---------------------------------------------------------------------------
# Cancel orders
# ---------------------------------------------------------------------------

class TestCancelOrders:

    def test_cancel_before_fill(self):
        book, engine, fills = fresh_engine()
        bid = make_limit("buyer", Side.BID, 100.0, 10)
        engine.process([bid])
        engine.process([make_cancel("buyer", bid.order_id)])

        # Now a sell market should find nothing
        engine.process([make_market("seller", Side.ASK, 5)])
        assert len(fills) == 0

    def test_cancel_partially_filled_order(self):
        """Fill half, then cancel remainder."""
        book, engine, fills = fresh_engine()
        bid = make_limit("buyer", Side.BID, 100.0, 10)
        engine.process([bid])
        # Fill 5
        engine.process([make_market("seller", Side.ASK, 5)])
        assert fills[-1].qty == 5
        assert bid.remaining_qty == 5
        # Cancel the remainder
        engine.process([make_cancel("buyer", bid.order_id)])
        # Another sell finds nothing
        engine.process([make_market("seller2", Side.ASK, 5)])
        assert len([f for f in fills if f.buy_agent_id == "buyer"]) == 1
