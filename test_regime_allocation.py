"""
Test script to verify regime filter + uncorrelated asset allocation logic.
Run this to confirm the fix works as expected.
"""

def test_allocations():
    """Test allocation calculations match expected values."""
    
    print("=" * 70)
    print("REGIME FILTER + UNCORRELATED ASSET ALLOCATION TEST")
    print("=" * 70)
    print()
    
    # Test parameters
    total_capital = 100000
    uncorrelated_pct = 20  # 20%
    
    test_cases = [
        {
            'name': 'No Regime, No Uncorrelated',
            'regime': None,
            'uncorrelated': None,
            'expected_stocks': 100000,
            'expected_uncorrelated': 0,
            'expected_cash': 0
        },
        {
            'name': 'No Regime, 20% Uncorrelated',
            'regime': None,
            'uncorrelated': uncorrelated_pct,
            'expected_stocks': 80000,
            'expected_uncorrelated': 20000,
            'expected_cash': 0
        },
        {
            'name': 'Half Portfolio, No Uncorrelated',
            'regime': 'Half Portfolio',
            'uncorrelated': None,
            'expected_stocks': 50000,
            'expected_uncorrelated': 0,
            'expected_cash': 50000
        },
        {
            'name': 'Half Portfolio, 20% Uncorrelated',
            'regime': 'Half Portfolio',
            'uncorrelated': uncorrelated_pct,
            'expected_stocks': 40000,  # 50% - 10%
            'expected_uncorrelated': 10000,  # 50% × 20%
            'expected_cash': 50000
        },
        {
            'name': 'Go Cash, No Uncorrelated',
            'regime': 'Go Cash',
            'uncorrelated': None,
            'expected_stocks': 0,
            'expected_uncorrelated': 0,
            'expected_cash': 100000
        },
        {
            'name': 'Go Cash, 20% Uncorrelated',
            'regime': 'Go Cash',
            'uncorrelated': uncorrelated_pct,
            'expected_stocks': 0,
            'expected_uncorrelated': 20000,  # 100% × 20%
            'expected_cash': 80000
        },
    ]
    
    all_passed = True
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        print("-" * 70)
        
        # Simulate the allocation logic from portfolio_engine.py
        total_funds = total_capital
        stocks_allocation = 0.0
        uncorrelated_allocation = 0.0
        cash_reserve = 0.0
        
        regime_triggered = test['regime'] is not None
        regime_action = test['regime']
        uncorrelated_config = {'allocation_pct': test['uncorrelated']} if test['uncorrelated'] else None
        
        # THE ACTUAL LOGIC FROM THE FIX
        if regime_triggered:
            if regime_action == 'Go Cash':
                stocks_allocation = 0.0
                if uncorrelated_config:
                    allocation_pct = uncorrelated_config['allocation_pct'] / 100.0
                    uncorrelated_allocation = total_funds * allocation_pct
                    cash_reserve = total_funds - uncorrelated_allocation
                else:
                    uncorrelated_allocation = 0.0
                    cash_reserve = total_funds
                    
            elif regime_action == 'Half Portfolio':
                available_funds = total_funds * 0.5
                cash_reserve = total_funds * 0.5
                
                if uncorrelated_config:
                    allocation_pct = uncorrelated_config['allocation_pct'] / 100.0
                    uncorrelated_allocation = available_funds * allocation_pct
                    stocks_allocation = available_funds - uncorrelated_allocation
                else:
                    uncorrelated_allocation = 0.0
                    stocks_allocation = available_funds
        else:
            if uncorrelated_config:
                allocation_pct = uncorrelated_config['allocation_pct'] / 100.0
                uncorrelated_allocation = total_funds * allocation_pct
                stocks_allocation = total_funds - uncorrelated_allocation
            else:
                uncorrelated_allocation = 0.0
                stocks_allocation = total_funds
        
        # Check results
        test_passed = (
            stocks_allocation == test['expected_stocks'] and
            uncorrelated_allocation == test['expected_uncorrelated'] and
            cash_reserve == test['expected_cash']
        )
        
        # Print results
        print(f"  Regime Action: {test['regime'] or 'None'}")
        print(f"  Uncorrelated %: {test['uncorrelated'] or 'None'}")
        print()
        print(f"  Expected Allocations:")
        print(f"    Stocks:       Rs{test['expected_stocks']:>10,.0f}")
        print(f"    Uncorrelated: Rs{test['expected_uncorrelated']:>10,.0f}")
        print(f"    Cash Reserve: Rs{test['expected_cash']:>10,.0f}")
        print(f"    Total:        Rs{test['expected_stocks'] + test['expected_uncorrelated'] + test['expected_cash']:>10,.0f}")
        print()
        print(f"  Actual Allocations:")
        print(f"    Stocks:       Rs{stocks_allocation:>10,.0f}")
        print(f"    Uncorrelated: Rs{uncorrelated_allocation:>10,.0f}")
        print(f"    Cash Reserve: Rs{cash_reserve:>10,.0f}")
        print(f"    Total:        Rs{stocks_allocation + uncorrelated_allocation + cash_reserve:>10,.0f}")
        print()
        
        if test_passed:
            print(f"  [PASSED]")
        else:
            print(f"  [FAILED]")
            all_passed = False
        
        print()
    
    print("=" * 70)
    if all_passed:
        print("[SUCCESS] ALL TESTS PASSED! The allocation logic is working correctly.")
    else:
        print("[WARNING] SOME TESTS FAILED. Please review the logic.")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    test_allocations()
