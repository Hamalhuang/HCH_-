import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, ttest_1samp
from scipy import stats
from IPython.display import display
warnings.filterwarnings('ignore')
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
class SingleFactorTool:
    """
    单因子测试工具类（适配你的 merged_df 结构）
    必需数据列：['date','order_book_id','factor_value','close']
    使用前请确保 merged_df 已包含 future_return（或使用 forward_days 参数生成）
    """

    def __init__(self, factor_long_df: pd.DataFrame, price_df: pd.DataFrame = None):
        """
        factor_long_df: 因子长表，列名应包含 ['date','order_book_id','factor_value']
        price_df: 可选，包含 ['date','order_book_id','close','industry_name'(可选)]
        """
        self.factor = factor_long_df.copy()
        if price_df is not None:
            self.price = price_df.copy()
            # 合并
            self.merged = pd.merge(self.factor, self.price, on=['date', 'order_book_id'], how='inner')
        else:
            self.price = None
            self.merged = self.factor.copy()

        # 统一类型
        if 'date' in self.merged.columns:
            self.merged['date'] = pd.to_datetime(self.merged['date'])
        # 如果没有 future_return，可通过 generate_future_return 生成
        if 'future_return' not in self.merged.columns and 'close' in self.merged.columns:
            self.merged['future_return'] = np.nan

    def generate_future_return(self, forward_days: int = 1):
        """基于 close 计算 future_return（group by order_book_id）"""
        if 'close' not in self.merged.columns:
            raise ValueError("缺少 close 列，无法计算 future_return")
        self.merged = self.merged.sort_values(['order_book_id', 'date'])
        self.merged['future_return'] = self.merged.groupby('order_book_id')['close'].shift(-forward_days) / self.merged['close'] - 1
        return self.merged

    def calculate_ic_cross_sectional(self, use_spearman: bool = True):
        """
        计算横截面 IC 时间序列与统计量
        返回 (ic_df, stats_dict)
        """
        ic_series = []
        for date, g in self.merged.groupby('date'):
            g = g.dropna(subset=['factor_value', 'future_return'])
            if len(g) > 1:
                if use_spearman:
                    corr, _ = spearmanr(g['factor_value'], g['future_return'])
                else:
                    corr = g['factor_value'].corr(g['future_return'])
                ic_series.append({'date': date, 'ic': corr})
        ic_df = pd.DataFrame(ic_series).sort_values('date')
        if ic_df.empty:
            stats_out = {k: np.nan for k in ['IC_mean', 'IC_std', 'ICIR', 'IC_positive_ratio', 'IC_skew', 'IC_kurtosis', 'IC_tvalue', 'IC_pvalue']}
            return ic_df, stats_out

        ic_vals = ic_df['ic'].astype(float).dropna()
        ic_mean = ic_vals.mean()
        ic_std = ic_vals.std(ddof=1)
        icir = ic_mean / ic_std if ic_std != 0 else np.nan
        ic_positive_ratio = (ic_vals > 0).mean()
        ic_skew = stats.skew(ic_vals, nan_policy='omit')
        ic_kurtosis = stats.kurtosis(ic_vals, fisher=True, nan_policy='omit')
        t_stat, p_value = ttest_1samp(ic_vals, 0, nan_policy='omit')

        stats_out = {
            'IC_mean': ic_mean,
            'IC_std': ic_std,
            'ICIR': icir,
            'IC_positive_ratio': ic_positive_ratio,
            'IC_skew': ic_skew,
            'IC_kurtosis': ic_kurtosis,
            'IC_tvalue': t_stat,
            'IC_pvalue': p_value
        }

        return ic_df, stats_out

    def plot_ic(self, ic_df, figsize=(14,5)):
        """整齐绘制 IC 时间序列与累积 IC"""
        if ic_df.empty:
            print("无 IC 数据可绘制")
            return
        ic_df = ic_df.copy()
        ic_df['date'] = pd.to_datetime(ic_df['date'])
        ic_df = ic_df.sort_values('date').set_index('date')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ic_df['ic'].plot(ax=ax1, title='IC Time Series', legend=False)
        ax1.set_ylabel('IC')
        ax1.set_xlabel('Date')
        ax1.grid(True)

        ic_df['ic'].cumsum().plot(ax=ax2, title='Cumulative IC Curve', color='orange', legend=False)
        ax2.set_ylabel('Cumulative IC')
        ax2.set_xlabel('Date')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def create_quantile_groups(self, n_groups: int = 5, inplace: bool = True):
        """每日期按因子分组，返回带 group 列的 DataFrame"""
        df = self.merged if inplace else self.merged.copy()
        df = df.copy()
        df['group'] = df.groupby('date')['factor_value'].transform(
            lambda x: pd.qcut(x, q=n_groups, labels=False, duplicates='drop')
        )
        if inplace:
            self.merged = df
        return df

    def layered_effect_analysis(self, n_groups: int = 5, return_col: str = 'future_return', plot: bool = True):
        """
        分层分析：返回每组日均收益序列、分组净值、多空净值（top-bottom）
        返回 dict 包含 group_daily_return, group_nav, long_short_return, long_short_nav
        """
        df = self.merged.copy()
        if return_col not in df.columns:
            raise ValueError(f"缺少收益列 {return_col}")
        df = self.create_quantile_groups(n_groups=n_groups, inplace=False)
        group_daily_return = df.groupby(['date', 'group'])[return_col].mean().unstack()
        group_nav = (1 + group_daily_return).cumprod()
        long_short_return = group_daily_return[n_groups - 1] - group_daily_return[0]
        long_short_nav = (1 + long_short_return).cumprod()

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            group_nav.plot(ax=axes[0], legend=True, title='Group NAV (Equal weight per group)')
            axes[0].set_ylabel('NAV')
            axes[0].set_xlabel('Date')
            axes[0].grid(True)

            long_short_nav.plot(ax=axes[1], color='red', title='Long-Short NAV (Top - Bottom)', legend=False)
            axes[1].set_ylabel('NAV')
            axes[1].set_xlabel('Date')
            axes[1].grid(True)

            plt.tight_layout()
            plt.show()

        return {
            'group_daily_return': group_daily_return,
            'group_nav': group_nav,
            'long_short_return': long_short_return,
            'long_short_nav': long_short_nav
        }

    def industry_exposure(self, industry_col: str = 'industry_name'):
        """
        计算各行业因子暴露度 (industry_mean - market_mean) / market_std
        需要 merged 含 industry 列
        返回 Series（按行业索引）
        """
        if industry_col not in self.merged.columns:
            raise ValueError(f"缺少行业列: {industry_col}")
        f = self.merged.dropna(subset=['factor_value'])
        market_mean = f['factor_value'].mean()
        market_std = f['factor_value'].std(ddof=1)
        industry_mean = f.groupby(industry_col)['factor_value'].mean()
        exposure = (industry_mean - market_mean) / (market_std if market_std != 0 else np.nan)
        return exposure.sort_values()

    def plot_industry_exposure(self, exposure_series: pd.Series, figsize=(10,6), color='skyblue'):
        plt.figure(figsize=figsize)
        exposure_series.plot(kind='barh', color=color)
        plt.title('Industry Factor Exposure (f-m)/u')
        plt.xlabel('Exposure (std units)')
        plt.ylabel('Industry')
        plt.grid(axis='x')
        plt.tight_layout()
        plt.show()

    def factor_hit_rates(self, n_groups: int = 5):
        """
        计算三种胜率并返回字典：
          - IC正率（按日 IC>0 的比例）
          - 股票层面方向胜率（平均每日因子与未来收益同号比例）
          - 多空胜率（按日 Top mean > Bottom mean 的比例）
        """
        df = self.merged.dropna(subset=['factor_value', 'future_return']).copy()
        if df.empty:
            print("factor_hit_rates: 没有可用的因子/未来收益数据，请先运行 generate_future_return() 或检查数据。")
            return {
                'IC_positive_ratio': np.nan,
                'stock_directional_hit_rate': np.nan,
                'long_short_win_rate': np.nan,
                'daily_stock_hit_series': pd.Series(dtype=float)
            }
        # IC 正率
        ic_list = []
        for date, g in df.groupby('date'):
            if len(g) > 1:
                ic, _ = spearmanr(g['factor_value'], g['future_return'])
                ic_list.append(ic)
        ic_list = pd.Series(ic_list)
        ic_positive_ratio = (ic_list > 0).mean() if not ic_list.empty else np.nan

        # 股票层面方向胜率（每日平均）
        df['hit'] = (df['factor_value'] * df['future_return']) > 0
        daily_stock_hit = df.groupby('date')['hit'].mean()
        stock_directional_hit_rate = daily_stock_hit.mean() if not daily_stock_hit.empty else np.nan
        # 显式绘图，确保在 Notebook 中显示
        fig, ax = plt.subplots(figsize=(10, 3))
        if not daily_stock_hit.empty:
            daily_stock_hit.plot(ax=ax, title='Daily Stock Directional Hit Rate', legend=False)
            ax.set_ylabel('Hit Rate')
            ax.set_xlabel('Date')
            ax.grid(True)
            plt.tight_layout()
            display(fig)   # IPython 显示
            plt.show()     # 强制渲染
        else:
            print("daily_stock_hit 为空，未绘制图表。")
        # 多空胜率
        df2 = df.copy()
        df2['group'] = df2.groupby('date')['factor_value'].transform(
            lambda x: pd.qcut(x, q=n_groups, labels=False, duplicates='drop')
        )
        group_returns = df2.groupby(['date', 'group'])['future_return'].mean().unstack()
        valid = group_returns.dropna(subset=[0, n_groups - 1])
        top_bottom_win_rate = (valid[n_groups - 1] > valid[0]).mean() if not valid.empty else np.nan



        return {
            'IC_positive_ratio': ic_positive_ratio,
            'stock_directional_hit_rate': stock_directional_hit_rate,
            'long_short_win_rate': top_bottom_win_rate,
            'daily_stock_hit_series': daily_stock_hit
        }
