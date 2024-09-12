## Knowledge Based Filtering
- Recommendations not based on user's rating history, but on specific queries.
- Focus on ephemeral user needs rather than building persistent user model.
- Uses more complex knowledge structure
- Infrequently bought and highly customizable items:
- - cars
- - houses
- - financial services
- - jewelery
- Customers want to define their requirements explicitly.
- Core idea: just ask the user what he wants step by step.
- Examples: 
- - House: price, number of rooms, garden or not, detached or not, etc.
- - Car: price, brand, color, fuel type, etc.
- Usually based on IF-THEN Business Rules:
- - Examples: if car = sport car then fuel type = gasoline, if price < 20k euros then Porsche = no
- Hard vs Soft business rules
- Rank items according to weighted of satisfied constraints.

## Hybrid Filtering
- Combine multiple recommender systems: Best of all worlds, using e.g. a linear model.
- Based on assessment of when/for what users specific recommender systems work well or not.
- Examples: 
- - Offer non-personalized recommendations to new users until they have at least 10 ratings.
- - Combine 5 best items using user-user collaborative filtering with 5 best items using 
    item-item collaborative filtering.
- - Combine collaborative and content filtering.
