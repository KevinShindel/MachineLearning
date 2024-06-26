## Business value
- Sales assistance
- Improve user experience
- Help people find what they are looking for faster
- Finding interesting items that a user would not think of him/herself
- Customer retention
- Increased sales
- X-selling: 
- - Up-selling: recommend Westmalle to customer ordering Stella Artois
- - Cross-selling: recommend cheese to the customer ordering Westmalle
- - Down-selling: discourage more beers if customer had too many
- Increase "hit", "click through" and "lookers to bookers" rates

## Examples 
- Google recommendations restaurants relevant by user location.
- Amazon recommends products based on user's purchase history.
- Job recommendation system by LinkedIn (related by CV)
- Facebook recommends friends based on mutual friends.

## Impact
- 35% of Amazon sales come from recommendations
- Touching the void and Into Thin Air (book recommendations)
- 75% of what people watch on Netflix is from recommendations
- 60% of what people watch on YouTube is from recommendations
- Netflix saves $1 billion each year using RS.
- RS launched on October 2006 by Netflix

# Items & users
- RS are software tools and techniques providing suggestions for items to be of use to a user.
- RS problem statement:
- - Input: user characteristics (ratings, preferences, etc.), items characteristics (genre, price, etc.)
- - Output: Relevance score, Top N rankings, etc.

## Personalized versus unpersonalized recommendations

| Personalized                                                                            | Unpersonalized                                   |
|-----------------------------------------------------------------------------------------|--------------------------------------------------|
| Users get unique recommendations                                                        | Same recommendations for all users               |
| Based on user's past behavior                                                           | Based on general trends                          |
| More accurate                                                                           | Less accurate                                    |
| More complex                                                                            | Less complex                                     |
| Collaborative filtering, content filtering, knowledge-based filtering, hybrid filtering | Popularity-based filtering, average rating, etc. |

## RS Challenges

- Every person is uniq
- People don't know what they want
- Large data sets but limited data per user
- Cold-start problem: new users or items
- Accuracy, diversity, novelty, fairness, serendipity, scalability, etc.