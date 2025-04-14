from datetime import datetime
from math import exp
import numpy as np
from pandas import DataFrame
from freqtrade.constants import Config
from freqtrade.optimize.hyperopt import IHyperOptLoss

# 策略特定参数
TARGET_TRADES = 300  # 5分钟时间框架下的合理目标交易数
EXPECTED_MAX_PROFIT = 2.5  # 预期最大总收益率(250%)
MAX_ACCEPTED_TRADE_DURATION = 180  # 3小时(36根5分钟K线)
MAX_DRAWDOWN = -0.25  # 最大可接受回撤
WIN_RATE_THRESHOLD = 0.45  # 最低可接受胜率

class CofiBitHyperOptLoss(IHyperOptLoss):
    """
    CofiBit策略定制化损失函数
    优化目标：
    1. 平衡交易频率和收益
    2. 控制交易持续时间
    3. 考虑回撤和胜率
    4. 奖励稳定的盈利表现
    """

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Config,
        processed: dict[str, DataFrame],
        *args,
        **kwargs,
    ) -> float:
        """
        自定义损失函数，数值越小表示结果越好
        """
        if trade_count == 0:
            return float('inf')
            
        total_profit = results["profit_ratio"].sum()
        avg_profit = results["profit_ratio"].mean()
        trade_duration = results["trade_duration"].mean()
        
        # 计算关键指标
        winning_trades = len(results[results["profit_ratio"] > 0])
        win_rate = winning_trades / trade_count
        max_drawdown = results["profit_ratio"].cumsum().min() if not results.empty else 0
        profit_std = results["profit_ratio"].std()
        profit_factor = (results[results["profit_ratio"] > 0]["profit_ratio"].sum() / 
                        abs(results[results["profit_ratio"] < 0]["profit_ratio"].sum()))
        
        # 各损失分量
        trade_count_loss = 1 - 0.2 * exp(-((trade_count - TARGET_TRADES) ** 2) / 10**5.8)
        profit_loss = max(0, 1 - total_profit / EXPECTED_MAX_PROFIT)
        duration_loss = 0.3 * min(trade_duration / MAX_ACCEPTED_TRADE_DURATION, 1)
        drawdown_loss = 0.5 * max(0, (abs(max_drawdown) - abs(MAX_DRAWDOWN)) / abs(MAX_DRAWDOWN))
        win_rate_loss = 0.4 * max(0, (WIN_RATE_THRESHOLD - win_rate) / WIN_RATE_THRESHOLD)
        consistency_loss = 0.3 * (profit_std if not np.isnan(profit_std) else 1)
        
        # 组合损失函数
        total_loss = (
            trade_count_loss +
            profit_loss +
            duration_loss +
            drawdown_loss +
            win_rate_loss +
            consistency_loss
        )
        
        # 额外奖励条件
        if profit_factor > 1.5:
            total_loss *= 0.9
        if win_rate > 0.55:
            total_loss *= 0.85
        if max_drawdown > -0.15:
            total_loss *= 0.8
            
        return total_loss

    @staticmethod
    def calculate_sortino_ratio(results: DataFrame, min_acceptable_return=0.0):
        """
        计算Sortino比率(未在默认损失中使用，但可用于高级优化)
        """
        returns = results["profit_ratio"]
        downside_returns = returns[returns < min_acceptable_return]
        
        if len(downside_returns) < 1:
            return float('inf')
            
        expected_return = returns.mean()
        downside_std = downside_returns.std()
        
        return expected_return / downside_std if downside_std != 0 else float('inf')
