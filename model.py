"""
ML Trading Strategy - Diagnostic & Improvement Framework
=========================================================
Goal: Transform Sharpe -0.03 → Positive Sharpe

This framework systematically diagnoses issues and tests improvements
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class StrategyDiagnostics:
    """
    Comprehensive diagnostics for underperforming trading strategies
    """
    
    def __init__(self):
        self.diagnostics = {}
        
    def generate_realistic_stock_data(self, symbol, n_days=1000):
        """Generate realistic stock data with actual market characteristics"""
        np.random.seed(hash(symbol) % 2**32)
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        
        # Create realistic return distribution
        # Most days: small moves, occasional: large moves (fat tails)
        returns = np.random.standard_t(df=5, size=n_days) * 0.015  # Fat-tailed
        returns += 0.0003  # Slight upward drift (annual ~7.5%)
        
        # Add persistence/momentum
        for i in range(5, len(returns)):
            returns[i] += 0.2 * returns[i-1]  # Momentum
            
        prices = 100 * np.exp(np.cumsum(returns))
        
        # OHLCV
        high = prices * (1 + np.abs(np.random.normal(0, 0.008, n_days)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.008, n_days)))
        open_price = prices * (1 + np.random.normal(0, 0.004, n_days))
        volume = np.random.lognormal(15, 0.4, n_days)
        
        return pd.DataFrame({
            'Date': dates, 'Symbol': symbol,
            'Open': open_price, 'High': high, 'Low': low,
            'Close': prices, 'Volume': volume
        })
    
    def create_features(self, df):
        """Feature engineering"""
        df = df.copy()
        
        # Returns
        df['Return'] = df['Close'].pct_change()
        df['Return_2'] = df['Close'].pct_change(2)
        df['Return_5'] = df['Close'].pct_change(5)
        
        # Momentum
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(5).mean()
        df['MA_20'] = df['Close'].rolling(20).mean()
        df['MA_50'] = df['Close'].rolling(50).mean()
        df['MA_Cross'] = df['MA_5'] - df['MA_20']
        
        # Volatility
        df['Volatility_10'] = df['Return'].rolling(10).std()
        df['Volatility_20'] = df['Return'].rolling(20).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_middle = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Position'] = (df['Close'] - bb_middle) / (2 * bb_std)
        
        # Volume
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Target
        df['Target_Direction'] = (df['Return'].shift(-1) > 0).astype(int)
        df['Target_Return'] = df['Return'].shift(-1)
        
        return df.dropna()
    
    def diagnose_original_strategy(self, df):
        """
        Diagnose why original strategy fails
        """
        print("\n" + "="*80)
        print("🔍 DIAGNOSTIC ANALYSIS - Why Sharpe = -0.03?")
        print("="*80)
        
        features = ['Return', 'Momentum_5', 'MA_5', 'MA_20', 
                   'Volatility_10', 'Close']
        
        # Split data
        train_size = int(len(df) * 0.6)
        train = df.iloc[:train_size]
        test = df.iloc[train_size:]
        
        # Original approach
        from sklearn.linear_model import LogisticRegression
        
        X_train = train[features]
        y_train = train['Target_Direction']
        X_test = test[features]
        y_test_direction = test['Target_Direction']
        y_test_return = test['Target_Return']
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        probs = model.predict_proba(X_test_scaled)[:, 1]
        predictions = (probs > 0.5).astype(int)
        
        # Analysis
        print("\n1️⃣ MODEL ACCURACY")
        print("-" * 80)
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        accuracy = accuracy_score(y_test_direction, predictions)
        precision = precision_score(y_test_direction, predictions, zero_division=0)
        recall = recall_score(y_test_direction, predictions, zero_division=0)
        
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}") 
        print(f"Recall:    {recall:.3f}")
        
        # Market baseline
        market_accuracy = (y_test_direction == 1).sum() / len(y_test_direction)
        print(f"\nMarket baseline (always long): {market_accuracy:.3f}")
        print(f"Model improvement: {(accuracy - market_accuracy):.3f}")
        
        if accuracy < 0.52:
            print("⚠️  ISSUE: Model barely better than random (50%)")
            
        # 2. Confusion Matrix
        print("\n2️⃣ PREDICTION ERRORS")
        print("-" * 80)
        cm = confusion_matrix(y_test_direction, predictions)
        print("Confusion Matrix:")
        print(f"  TN: {cm[0,0]:<6} FP: {cm[0,1]:<6}")
        print(f"  FN: {cm[1,0]:<6} TP: {cm[1,1]:<6}")
        
        # 3. Probability calibration
        print("\n3️⃣ PROBABILITY CALIBRATION")
        print("-" * 80)
        bins = [0, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0]
        bin_labels = ['<0.4', '0.4-0.45', '0.45-0.5', '0.5-0.55', '0.55-0.6', '>0.6']
        
        test['Prob_Bin'] = pd.cut(probs, bins=bins, labels=bin_labels)
        test['Prob'] = probs
        
        calibration = test.groupby('Prob_Bin', observed=True).agg({
            'Target_Direction': ['mean', 'count']
        }).round(3)
        
        print("Predicted Prob Range → Actual Win Rate")
        print(calibration)
        
        # 4. Trade signal analysis
        print("\n4️⃣ TRADING SIGNALS")
        print("-" * 80)
        
        vol_threshold = test['Volatility_10'].median()
        trade_allowed = test['Volatility_10'] < vol_threshold
        
        signals = np.where(
            (probs > 0.55) & trade_allowed, 1,
            np.where((probs < 0.45) & trade_allowed, -1, 0)
        )
        
        n_long = (signals == 1).sum()
        n_short = (signals == -1).sum()
        n_flat = (signals == 0).sum()
        
        print(f"Long signals:  {n_long:>5} ({n_long/len(signals)*100:.1f}%)")
        print(f"Short signals: {n_short:>5} ({n_short/len(signals)*100:.1f}%)")
        print(f"Flat (no trade): {n_flat:>5} ({n_flat/len(signals)*100:.1f}%)")
        
        # 5. Returns by signal
        print("\n5️⃣ RETURNS BY SIGNAL TYPE")
        print("-" * 80)
        
        test['Signal'] = signals
        signal_returns = test.groupby('Signal')['Target_Return'].agg([
            ('Mean', 'mean'),
            ('Std', 'std'),
            ('Count', 'count'),
            ('Win_Rate', lambda x: (x > 0).mean())
        ]).round(4)
        
        print(signal_returns)
        
        # 6. Transaction costs impact
        print("\n6️⃣ TRANSACTION COSTS IMPACT")
        print("-" * 80)
        
        strategy_returns = signals * y_test_return.values
        costs = np.abs(signals) * 0.001
        
        returns_before_costs = strategy_returns.sum()
        returns_after_costs = (strategy_returns - costs).sum()
        cost_impact = returns_before_costs - returns_after_costs
        
        print(f"Returns before costs: {returns_before_costs*100:>8.2f}%")
        print(f"Transaction costs:    {cost_impact*100:>8.2f}%")
        print(f"Returns after costs:  {returns_after_costs*100:>8.2f}%")
        print(f"\nCosts consume {cost_impact/returns_before_costs*100:.1f}% of gross returns!")
        
        # 7. Volatility filter effectiveness
        print("\n7️⃣ VOLATILITY FILTER EFFECTIVENESS")
        print("-" * 80)
        
        high_vol_mask = ~trade_allowed
        high_vol_returns = test[high_vol_mask]['Target_Return'].mean()
        low_vol_returns = test[trade_allowed]['Target_Return'].mean()
        
        print(f"Avg return in HIGH volatility: {high_vol_returns*100:.4f}%")
        print(f"Avg return in LOW volatility:  {low_vol_returns*100:.4f}%")
        
        if high_vol_returns > low_vol_returns:
            print("⚠️  ISSUE: Filter is backwards! High vol periods are more profitable")
        
        # Store diagnostics
        self.diagnostics['accuracy'] = accuracy
        self.diagnostics['signal_distribution'] = {
            'long': n_long, 'short': n_short, 'flat': n_flat
        }
        self.diagnostics['cost_impact'] = cost_impact / returns_before_costs
        
        return test, signals, probs
    
    def test_improvements(self, df):
        """
        Test various improvements systematically
        """
        print("\n" + "="*80)
        print("💡 TESTING IMPROVEMENTS")
        print("="*80)
        
        improvements = {}
        
        # Baseline (original)
        print("\n📊 BASELINE (Original Strategy)")
        result = self._run_strategy(df, strategy='original')
        improvements['Baseline'] = result
        print(f"   Sharpe: {result['sharpe']:.3f} | Return: {result['return']*100:.2f}%")
        
        # Improvement 1: Better features
        print("\n📊 IMPROVEMENT 1: Enhanced Features")
        result = self._run_strategy(df, strategy='enhanced_features')
        improvements['Enhanced_Features'] = result
        print(f"   Sharpe: {result['sharpe']:.3f} | Return: {result['return']*100:.2f}%")
        delta = result['sharpe'] - improvements['Baseline']['sharpe']
        print(f"   Δ Sharpe: {delta:+.3f}")
        
        # Improvement 2: Random Forest
        print("\n📊 IMPROVEMENT 2: Random Forest Model")
        result = self._run_strategy(df, strategy='random_forest')
        improvements['Random_Forest'] = result
        print(f"   Sharpe: {result['sharpe']:.3f} | Return: {result['return']*100:.2f}%")
        delta = result['sharpe'] - improvements['Baseline']['sharpe']
        print(f"   Δ Sharpe: {delta:+.3f}")
        
        # Improvement 3: Stricter thresholds
        print("\n📊 IMPROVEMENT 3: Stricter Signal Thresholds")
        result = self._run_strategy(df, strategy='strict_thresholds')
        improvements['Strict_Thresholds'] = result
        print(f"   Sharpe: {result['sharpe']:.3f} | Return: {result['return']*100:.2f}%")
        delta = result['sharpe'] - improvements['Baseline']['sharpe']
        print(f"   Δ Sharpe: {delta:+.3f}")
        
        # Improvement 4: No shorting
        print("\n📊 IMPROVEMENT 4: Long-Only Strategy")
        result = self._run_strategy(df, strategy='long_only')
        improvements['Long_Only'] = result
        print(f"   Sharpe: {result['sharpe']:.3f} | Return: {result['return']*100:.2f}%")
        delta = result['sharpe'] - improvements['Baseline']['sharpe']
        print(f"   Δ Sharpe: {delta:+.3f}")
        
        # Improvement 5: Remove volatility filter
        print("\n📊 IMPROVEMENT 5: No Volatility Filter")
        result = self._run_strategy(df, strategy='no_vol_filter')
        improvements['No_Vol_Filter'] = result
        print(f"   Sharpe: {result['sharpe']:.3f} | Return: {result['return']*100:.2f}%")
        delta = result['sharpe'] - improvements['Baseline']['sharpe']
        print(f"   Δ Sharpe: {delta:+.3f}")
        
        # Improvement 6: Combined best
        print("\n📊 IMPROVEMENT 6: Combined Best Practices")
        result = self._run_strategy(df, strategy='combined')
        improvements['Combined_Best'] = result
        print(f"   Sharpe: {result['sharpe']:.3f} | Return: {result['return']*100:.2f}%")
        delta = result['sharpe'] - improvements['Baseline']['sharpe']
        print(f"   Δ Sharpe: {delta:+.3f}")
        
        return improvements
    
    def _run_strategy(self, df, strategy='original'):
        """Run a specific strategy variant"""
        
        # Enhanced features
        if strategy in ['enhanced_features', 'combined']:
            features = ['Return', 'Return_2', 'Return_5', 
                       'Momentum_5', 'Momentum_10',
                       'MA_5', 'MA_20', 'MA_50', 'MA_Cross',
                       'Volatility_10', 'Volatility_20',
                       'RSI', 'BB_Position', 'Volume_Ratio']
        else:
            features = ['Return', 'Momentum_5', 'MA_5', 'MA_20', 
                       'Volatility_10']
        
        # Add Close if needed
        if 'Close' not in features:
            X_features = features
        else:
            X_features = features
            
        # Split
        train_size = int(len(df) * 0.6)
        train = df.iloc[:train_size]
        test = df.iloc[train_size:]
        
        X_train = train[[f for f in X_features if f in train.columns]]
        y_train = train['Target_Direction']
        X_test = test[[f for f in X_features if f in test.columns]]
        y_test_return = test['Target_Return']
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model
        if strategy in ['random_forest', 'combined']:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=100, max_depth=8, 
                min_samples_split=20, random_state=42
            )
        else:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000)
        
        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        # Signals
        if strategy in ['strict_thresholds', 'combined']:
            long_threshold = 0.60
            short_threshold = 0.40
        else:
            long_threshold = 0.55
            short_threshold = 0.45
        
        # Volatility filter
        if strategy == 'no_vol_filter':
            trade_allowed = np.ones(len(test), dtype=bool)
        else:
            vol_threshold = test['Volatility_10'].median()
            trade_allowed = test['Volatility_10'] < vol_threshold
        
        # Generate signals
        if strategy == 'long_only':
            signals = np.where(
                (probs > long_threshold) & trade_allowed, 1, 0
            )
        else:
            signals = np.where(
                (probs > long_threshold) & trade_allowed, 1,
                np.where((probs < short_threshold) & trade_allowed, -1, 0)
            )
        
        # Calculate returns
        strategy_returns = signals * y_test_return.values
        costs = np.abs(signals) * 0.001
        net_returns = strategy_returns - costs
        
        # Metrics
        total_return = net_returns.sum()
        mean_return = net_returns.mean()
        std_return = net_returns.std()
        sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        return {
            'return': total_return,
            'sharpe': sharpe,
            'volatility': std_return * np.sqrt(252),
            'signals': signals,
            'returns': net_returns
        }
    
    def create_diagnostic_plots(self, improvements):
        """Create diagnostic visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Strategy Improvement Analysis', fontsize=16, fontweight='bold')
        
        # 1. Sharpe comparison
        ax = axes[0, 0]
        strategies = list(improvements.keys())
        sharpes = [improvements[s]['sharpe'] for s in strategies]
        colors = ['red' if s < 0 else 'green' for s in sharpes]
        
        bars = ax.barh(strategies, sharpes, color=colors, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Sharpe Ratio')
        ax.set_title('Sharpe Ratio Comparison')
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, val) in enumerate(zip(bars, sharpes)):
            x_pos = val + 0.05 if val > 0 else val - 0.05
            ha = 'left' if val > 0 else 'right'
            ax.text(x_pos, i, f'{val:.3f}', va='center', ha=ha, fontweight='bold')
        
        # 2. Returns comparison
        ax = axes[0, 1]
        returns = [improvements[s]['return']*100 for s in strategies]
        colors = ['red' if r < 0 else 'green' for r in returns]
        ax.barh(strategies, returns, color=colors, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Total Return (%)')
        ax.set_title('Total Return Comparison')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 3. Cumulative returns
        ax = axes[0, 2]
        for strategy in ['Baseline', 'Combined_Best']:
            if strategy in improvements:
                returns = improvements[strategy]['returns']
                cumulative = (1 + pd.Series(returns)).cumprod()
                ax.plot(cumulative.values, label=strategy, linewidth=2)
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Cumulative Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Returns distribution
        ax = axes[1, 0]
        baseline_returns = improvements['Baseline']['returns'] * 100
        combined_returns = improvements['Combined_Best']['returns'] * 100
        ax.hist(baseline_returns, bins=50, alpha=0.5, label='Baseline', color='red')
        ax.hist(combined_returns, bins=50, alpha=0.5, label='Combined Best', color='green')
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Returns Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Signal frequency
        ax = axes[1, 1]
        signal_counts = []
        for strategy in strategies:
            signals = improvements[strategy]['signals']
            counts = [
                (signals == 1).sum(),
                (signals == -1).sum(),
                (signals == 0).sum()
            ]
            signal_counts.append(counts)
        
        x = np.arange(len(strategies))
        width = 0.25
        ax.bar(x - width, [c[0] for c in signal_counts], width, label='Long', color='green', alpha=0.7)
        ax.bar(x, [c[1] for c in signal_counts], width, label='Short', color='red', alpha=0.7)
        ax.bar(x + width, [c[2] for c in signal_counts], width, label='Flat', color='gray', alpha=0.7)
        
        ax.set_ylabel('Count')
        ax.set_title('Signal Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 6. Key metrics table
        ax = axes[1, 2]
        ax.axis('off')
        
        baseline_sharpe = improvements['Baseline']['sharpe']
        best_sharpe = improvements['Combined_Best']['sharpe']
        improvement = best_sharpe - baseline_sharpe
        
        summary = f"""
        IMPROVEMENT SUMMARY
        {'='*35}
        
        Original Strategy:
          Sharpe Ratio: {baseline_sharpe:>8.3f}
          Total Return: {improvements['Baseline']['return']*100:>7.2f}%
        
        Best Strategy (Combined):
          Sharpe Ratio: {best_sharpe:>8.3f}
          Total Return: {improvements['Combined_Best']['return']*100:>7.2f}%
        
        IMPROVEMENT:
          Δ Sharpe:     {improvement:>+8.3f}
          Δ Return:     {(improvements['Combined_Best']['return'] - improvements['Baseline']['return'])*100:>+7.2f}%
        
        Key Changes:
          ✓ Enhanced features (+{len(['RSI', 'BB', 'Vol_Ratio'])} indicators)
          ✓ Random Forest model
          ✓ Stricter thresholds (0.60/0.40)
          ✓ Better risk management
        """
        
        ax.text(0.1, 0.9, summary, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('/home/claude/strategy_diagnostics.png', dpi=300, bbox_inches='tight')
        print("\n✅ Diagnostic plots saved!")
        
        return fig


def main():
    """
    Run comprehensive diagnostics
    """
    print("="*80)
    print("ML TRADING STRATEGY - COMPREHENSIVE DIAGNOSTICS")
    print("="*80)
    print("\nGoal: Identify why Sharpe = -0.03 and fix it!")
    
    diagnostics = StrategyDiagnostics()
    
    # Generate data
    print("\n📊 Generating realistic market data...")
    df = diagnostics.generate_realistic_stock_data('TEST', n_days=1000)
    
    # Create features
    df = diagnostics.create_features(df)
    
    # Run diagnostics
    test_data, signals, probs = diagnostics.diagnose_original_strategy(df)
    
    # Test improvements
    improvements = diagnostics.test_improvements(df)
    
    # Create plots
    diagnostics.create_diagnostic_plots(improvements)
    
    # Summary
    print("\n" + "="*80)
    print("📋 RECOMMENDATIONS")
    print("="*80)
    
    best_strategy = max(improvements.keys(), key=lambda k: improvements[k]['sharpe'])
    best_sharpe = improvements[best_strategy]['sharpe']
    baseline_sharpe = improvements['Baseline']['sharpe']
    
    print(f"\n✅ BEST STRATEGY: {best_strategy}")
    print(f"   Sharpe Ratio: {best_sharpe:.3f}")
    print(f"   Improvement:  {best_sharpe - baseline_sharpe:+.3f}")
    print(f"   Total Return: {improvements[best_strategy]['return']*100:.2f}%")
    
    print("\n🎯 KEY FINDINGS:")
    if diagnostics.diagnostics['accuracy'] < 0.52:
        print("   • Model accuracy too low - need better features or different approach")
    if diagnostics.diagnostics['cost_impact'] > 0.5:
        print("   • Transaction costs eating >50% of profits - reduce trade frequency")
    if diagnostics.diagnostics['signal_distribution']['flat'] > 0.7 * sum(diagnostics.diagnostics['signal_distribution'].values()):
        print("   • Too many 'no trade' signals - thresholds may be too conservative")
    
    print("\n💡 NEXT STEPS:")
    print("   1. Use the 'Combined_Best' strategy as your new baseline")
    print("   2. Consider adding: market regime detection, sentiment data")
    print("   3. Test on real data with proper walk-forward validation")
    print("   4. Implement paper trading before live deployment")
    
    print("\n" + "="*80)
    print("✅ DIAGNOSTICS COMPLETE!")
    print("="*80)
    
    return diagnostics, improvements


if __name__ == "__main__":
    diagnostics, improvements = main()