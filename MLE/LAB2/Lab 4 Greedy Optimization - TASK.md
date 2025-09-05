Q: **Change the code to have a flag for dollars and flag for cents, i.e., --dollars and --cents.**

A: **To implement the flags for dollars and cents, we'll need to modify the code using the library `click`**
```python
import click

@click.command()
@click.option('--dollars', type=float, default=0.0, help='Amount in dollars')
@click.option('--cents', type=int, default=0, help='Amount in cents')
def make_change(dollars, cents):
    """Calculate minimum coins needed for the given amount."""
    # Convert everything to cents for calculation
    total_cents = int(dollars * 100) + cents
    
    if total_cents < 0:
        click.echo("Error: Amount cannot be negative")
        return
    
    coins = [25, 10, 5, 1]  # Quarter, dime, nickel, penny
    result = []
    remaining = total_cents
    
    for coin in coins:
        count = remaining // coin
        if count > 0:
            result.append((coin, count))
            remaining -= count * coin
    
    # Format output
    click.echo(f"Change for ${dollars + cents/100:.2f}:")
    for coin, count in result:
        coin_name = {
            25: "quarter(s)",
            10: "dime(s)",
            5: "nickel(s)",
            1: "penny/pennies"
        }[coin]
        click.echo(f"{count} {coin_name}")

if __name__ == '__main__':
    make_change()
```

Q: **Reflection question: Is this version of the command-line tool more robust against errors?**

A: **Regarding robustness, this version is indeed more robust for several reasons:**
- It separates dollars and cents clearly, reducing input errors
- It uses Click's built-in type checking for input validation
- It handles negative numbers explicitly
- It provides clear help messages through Click's documentation features

Q: **Reflection question: What could you build to enhance this script? Do it and add it to your portfolio.**

A: **Here are several valuable additions**
```python
import click
import json
from datetime import datetime
from pathlib import Path

class CoinCalculator:
    def __init__(self):
        self.coin_systems = {
            'usd': [25, 10, 5, 1],
            'eur': [200, 100, 50, 20, 10, 5, 2, 1],
            'gbp': [200, 100, 50, 20, 10, 5, 2, 1]
        }
        self.history_file = Path('change_history.json')

    def calculate_change(self, amount_cents, currency='usd'):
        coins = self.coin_systems.get(currency.lower(), self.coin_systems['usd'])
        result = []
        remaining = amount_cents
        
        for coin in coins:
            count = remaining // coin
            if count > 0:
                result.append((coin, count))
                remaining -= count * coin
        return result

    def save_transaction(self, amount, result, currency='usd'):
        history = []
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                history = json.load(f)
        
        history.append({
            'timestamp': datetime.now().isoformat(),
            'amount': amount,
            'currency': currency,
            'change': [(coin, count) for coin, count in result]
        })
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

@click.group()
def cli():
    """Enhanced coin change calculator with multiple currencies and history."""
    pass

@cli.command()
@click.option('--dollars', type=float, default=0.0, help='Amount in dollars')
@click.option('--cents', type=int, default=0, help='Amount in cents')
@click.option('--currency', type=click.Choice(['USD', 'EUR', 'GBP'], case_sensitive=False), default='USD')
def calculate(dollars, cents, currency):
    """Calculate minimum coins needed for the given amount."""
    calculator = CoinCalculator()
    total_cents = int(dollars * 100) + cents
    
    if total_cents < 0:
        click.echo("Error: Amount cannot be negative")
        return
    
    result = calculator.calculate_change(total_cents, currency.lower())
    calculator.save_transaction(total_cents/100, result, currency)
    
    click.echo(f"Change for {currency} ${dollars + cents/100:.2f}:")
    for coin, count in result:
        click.echo(f"{count} x {coin/100:.2f} {currency}")

@cli.command()
def history():
    """Show calculation history."""
    calculator = CoinCalculator()
    if not calculator.history_file.exists():
        click.echo("No history available")
        return
    
    with open(calculator.history_file, 'r') as f:
        history = json.load(f)
    
    for entry in history:
        click.echo(f"\nDate: {entry['timestamp']}")
        click.echo(f"Amount: {entry['currency']} ${entry['amount']:.2f}")
        click.echo("Change:")
        for coin, count in entry['change']:
            click.echo(f"  {count} x {coin/100:.2f} {entry['currency']}")

if __name__ == '__main__':
    cli()
```

- Multiple currency support
- Transaction history
- Command-line subcommands
- Better error handling
- JSON storage for transaction history
- Modular design with a separate calculator class


Q: **What is the optimal number of simulations to run?**

A: **Generally:**
- For small problems (< 20 cities): 1000-10000 simulations might be sufficient
- For medium problems (20-50 cities): 10000-100000 simulations

We can determine the optimal number by:
1. Running increasing numbers of simulations
2. Plotting the best route length vs number of simulations
3. Looking for the point where additional simulations don't significantly improve results
4. Considering your time/accuracy tradeoff requirements
