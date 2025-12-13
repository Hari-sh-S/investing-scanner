import re
import pandas as pd
import numpy as np

class ScoreParser:
    def __init__(self):
        # Expanded metrics list with all new additions
        self.allowed_metrics = [
            # Performance
            '1 Month Performance', '3 Month Performance', '6 Month Performance', 
            '9 Month Performance', '1 Year Performance',
            # Volatility
            '1 Month Volatility', '3 Month Volatility', '6 Month Volatility', 
            '9 Month Volatility', '1 Year Volatility',
            # Max Drawdown
            '1 Month Max Drawdown', '3 Month Max Drawdown', '6 Month Max Drawdown',
            '9 Month Max Drawdown', '1 Year Max Drawdown',
            # Sharpe Ratio
            '1 Month Sharpe', '3 Month Sharpe', '6 Month Sharpe',
            '9 Month Sharpe', '1 Year Sharpe',
            # Sortino Ratio
            '1 Month Sortino', '3 Month Sortino', '6 Month Sortino',
            '9 Month Sortino', '1 Year Sortino',
            # Calmar Ratio
            '1 Month Calmar', '3 Month Calmar', '6 Month Calmar',
            '9 Month Calmar', '1 Year Calmar',
        ]
        
        # Create metric groups for UI display
        self.metric_groups = {
            'Performance': ['1 Month Performance', '3 Month Performance', '6 Month Performance', 
                          '9 Month Performance', '1 Year Performance'],
            'Volatility': ['1 Month Volatility', '3 Month Volatility', '6 Month Volatility',
                         '9 Month Volatility', '1 Year Volatility'],
            'Max Drawdown': ['1 Month Max Drawdown', '3 Month Max Drawdown', '6 Month Max Drawdown',
                           '9 Month Max Drawdown', '1 Year Max Drawdown'],
            'Sharpe Ratio': ['1 Month Sharpe', '3 Month Sharpe', '6 Month Sharpe',
                           '9 Month Sharpe', '1 Year Sharpe'],
            'Sortino Ratio': ['1 Month Sortino', '3 Month Sortino', '6 Month Sortino',
                            '9 Month Sortino', '1 Year Sortino'],
            'Calmar Ratio': ['1 Month Calmar', '3 Month Calmar', '6 Month Calmar',
                           '9 Month Calmar', '1 Year Calmar'],
        }

    def validate_formula(self, formula):
        """
        Validates the formula before execution.
        Returns (is_valid, error_message)
        """
        if not formula or formula.strip() == "":
            return False, "Formula cannot be empty"
        
        # Check for balanced parentheses
        if formula.count('(') != formula.count(')'):
            return False, "Unbalanced parentheses"
        
        # Preprocess
        processed = self._preprocess_formula(formula)
        
        # Replace all known metrics with (1) for testing
        # Sort by length (longest first) to avoid partial replacements
        test_formula = processed
        for metric in sorted(self.allowed_metrics, key=len, reverse=True):
            # Use exact replacement to avoid issues
            if metric in test_formula:
                test_formula = test_formula.replace(metric, "(1)")
        
        # Check if only valid characters remain
        # Remove all numbers, operators, parentheses, and spaces
        test_clean = re.sub(r'[\d\.\+\-\*\/\(\)\s]', '', test_formula)
        if test_clean:
            return False, f"Unknown: {test_clean[:15]}"
        
        # Try to evaluate with dummy values to check syntax
        try:
            result = eval(test_formula)
            if result is None:
                return False, "Returns None"
            return True, "Valid"
        except ZeroDivisionError:
            # Division by zero is okay during validation
            return True, "Valid"
        except SyntaxError as e:
            return False, f"Syntax error"
        except Exception as e:
            return False, f"Error: {str(e)[:25]}"
    
    def _preprocess_formula(self, formula):
        """Internal method to preprocess formula."""
        processed = formula
        
        # Replace percentages (e.g., 80% -> 0.80)
        processed = re.sub(r'(\d+(\.\d+)?)%', r'(\1/100)', processed)
        
        # Replace operators
        processed = processed.replace('x', '*').replace('×', '*')
        processed = processed.replace('÷', '/')
        
        return processed

    def parse_and_calculate(self, formula, row):
        """
        Parses and evaluates formula for a single row.
        Returns the score or a very negative number on error.
        """
        processed_formula = self._preprocess_formula(formula)
        
        # Replace metrics with values from row
        for metric in sorted(self.allowed_metrics, key=len, reverse=True):  # Longest first
            if metric in processed_formula:
                val = row.get(metric, 0)
                
                # Handle Series case (convert to scalar)
                if isinstance(val, pd.Series):
                    val = val.iloc[0] if len(val) > 0 else 0
                
                # Handle NaN and inf
                if pd.isna(val):
                    val = 0
                elif np.isinf(val):
                    val = 0
                    
                processed_formula = processed_formula.replace(metric, str(val))
        
        # Final safety check
        if not re.match(r'^[\d\.\+\-\*\/\(\)\s]+$', processed_formula):
            return -999999
            
        try:
            score = eval(processed_formula)
            if np.isnan(score) or np.isinf(score):
                return -999999
            return score
        except Exception as e:
            return -999999

    def calculate_scores(self, df, formula):
        """
        Calculate scores for entire dataframe using row-by-row processing.
        Returns a Series of scores.
        """
        # Use reliable row-by-row calculation
        return df.apply(lambda row: self.parse_and_calculate(formula, row), axis=1)
    
    def get_example_formulas(self):
        """Returns a dictionary of example formulas with descriptions."""
        return {
            "Simple Momentum": "6 Month Performance",
            "Risk-Adjusted Momentum": "(6 Month Performance / 6 Month Volatility)",
            "Weighted Performance": "(70% * 6 Month Performance) + (20% * 3 Month Performance) + (10% * 1 Month Performance)",
            "Sharpe-Based": "6 Month Sharpe",
            "Multi-Factor": "((80% * 9 Month Performance) + (20% * 6 Month Performance)) / 1 Month Volatility",
            "Calmar Focus": "6 Month Calmar",
            "Low Volatility": "1 / 3 Month Volatility",
            "Momentum + Quality": "(6 Month Performance * 6 Month Sharpe)",
        }
