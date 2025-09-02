1. **Tool Operation Investigation:** The poker simulator has several key operations accessible through CLI commands:

- `info`: Shows all possible poker hands with their rankings (with optional probability display)
- : Simulates dealing poker hands `deal`
- `play`: Allows playing against computer with betting
- `interactive`: Provides a more detailed gaming experience with multiple rounds and statistics


**Five Improvements for Interactive Simulation:**
1. **Player Strategy Assistance:**
``` python
# Add probability-based suggestions for betting decisions
def suggest_bet(hand_probability, current_money):
    """Provides strategic betting advice based on hand probability"""
    if hand_probability > 40:  # High probability hand
        return "Consider a larger bet due to strong hand"
    return "Consider a conservative bet due to weaker hand"
```
2. **Add minimum/maximum bet limits**
3. **Allow raising/folding decisions**
4. **Add ASCII card visualization**
5. **Implement color-coded statistics**
