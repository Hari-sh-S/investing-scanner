"""
Debug script to test regime filter + uncorrelated asset allocation logic locally.
This will print detailed allocation information to help diagnose issues.
"""

# Simulate the allocation logic
def test_allocation(investable_capital, regime_triggered, regime_action, uncorrelated_pct):
    """Test allocation calculation."""
    
    print(f"\n{'='*70}")
    print(f"Test: Regime={regime_action if regime_triggered else 'None'}, Uncorrelated={uncorrelated_pct}%")
    print(f"Investable Capital: Rs {investable_capital:,.0f}")
    print(f"{'='*70}")
    
    stocks_target = 0.0
    uncorrelated_target = 0.0
    
    if regime_triggered:
        if regime_action == 'Go Cash':
            stocks_target = 0.0
            if uncorrelated_pct > 0:
                allocation_pct = uncorrelated_pct / 100.0
                uncorrelated_target = investable_capital * allocation_pct
                
        elif regime_action == 'Half Portfolio':
            # ALWAYS 50% to stocks
            stocks_target = investable_capital * 0.5
            if uncorrelated_pct > 0:
                allocation_pct = uncorrelated_pct / 100.0
                # Uncorrelated from the OTHER 50%
                uncorrelated_target = (investable_capital * 0.5) * allocation_pct
    else:
        # No regime
        if uncorrelated_pct > 0:
            allocation_pct = uncorrelated_pct / 100.0
            uncorrelated_target = investable_capital * allocation_pct
            stocks_target = investable_capital - uncorrelated_target
        else:
            uncorrelated_target = 0.0
            stocks_target = investable_capital
    
    cash_reserve = investable_capital - stocks_target - uncorrelated_target
    
    print(f"\nCalculated Allocations:")
    print(f"  Stocks Target:       Rs {stocks_target:>12,.0f} ({stocks_target/investable_capital*100:>5.1f}%)")
    print(f"  Uncorrelated Target: Rs {uncorrelated_target:>12,.0f} ({uncorrelated_target/investable_capital*100:>5.1f}%)")
    print(f"  Cash Reserve:        Rs {cash_reserve:>12,.0f} ({cash_reserve/investable_capital*100:>5.1f}%)")
    print(f"  Total:               Rs {stocks_target + uncorrelated_target + cash_reserve:>12,.0f}")
    
    return stocks_target, uncorrelated_target, cash_reserve


if __name__ == "__main__":
    capital = 100000
    
    print("\n" + "="*70)
    print("REGIME FILTER + UNCORRELATED ASSET ALLOCATION TEST")
    print("="*70)
    
    # Test 1: No regime, no uncorrelated
    test_allocation(capital, False, None, 0)
    
    # Test 2: No regime, 20% uncorrelated
    test_allocation(capital, False, None, 20)
    
    # Test 3: Half Portfolio, no uncorrelated
    test_allocation(capital, True, 'Half Portfolio', 0)
    
    # Test 4: Half Portfolio, 20% uncorrelated
    stocks, unc, cash = test_allocation(capital, True, 'Half Portfolio', 20)
    print(f"\nExpected: 50% stocks + 10% unc + 40% cash")
    print(f"Actual:   {stocks/capital*100:.0f}% stocks + {unc/capital*100:.0f}% unc + {cash/capital*100:.0f}% cash")
    assert stocks == 50000, f"Stocks should be 50000, got {stocks}"
    assert unc == 10000, f"Uncorrelated should be 10000, got {unc}"
    assert cash == 40000, f"Cash should be 40000, got {cash}"
    print("✓ PASS!")
    
    # Test 5: Half Portfolio, 100% uncorrelated
    stocks, unc, cash = test_allocation(capital, True, 'Half Portfolio', 100)
    print(f"\nExpected: 50% stocks + 50% unc + 0% cash")
    print(f"Actual:   {stocks/capital*100:.0f}% stocks + {unc/capital*100:.0f}% unc + {cash/capital*100:.0f}% cash")
    assert stocks == 50000, f"Stocks should be 50000, got {stocks}"
    assert unc == 50000, f"Uncorrelated should be 50000, got {unc}"
    assert cash == 0, f"Cash should be 0, got {cash}"
    print("✓ PASS!")
    
    # Test 6: Go Cash, no uncorrelated
    test_allocation(capital, True, 'Go Cash', 0)
    
    # Test 7: Go Cash, 20% uncorrelated
    stocks, unc, cash = test_allocation(capital, True, 'Go Cash', 20)
    print(f"\nExpected: 0% stocks + 20% unc + 80% cash")
    print(f"Actual:   {stocks/capital*100:.0f}% stocks + {unc/capital*100:.0f}% unc + {cash/capital*100:.0f}% cash")
    assert stocks == 0, f"Stocks should be 0, got {stocks}"
    assert unc == 20000, f"Uncorrelated should be 20000, got {unc}"
    assert cash == 80000, f"Cash should be 80000, got {cash}"
    print("✓ PASS!")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED! Allocation logic is correct.")
    print("="*70)
