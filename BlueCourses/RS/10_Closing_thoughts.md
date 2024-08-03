## Recommender Systems: Attacks
- Malicious user might be interested to increase or decrease rank of certain items
- Fake users which over or under rate target items:
- - Neighbourhood recommenders robust against this attack
- Fake item content:
- - Item is not what it is described to be
- Aim of attacks:
- - Push attack - increase predicted rating of target item
- - Nuke attack - decrease predicted rating of target item
- - Breakdown recommender system.
- Target of attacks:
- - User's
- - Items
- Type of attacks:
- - Random attack: Randomly rate items
- - Bandwagon attack: Rate items that are already popular
- - Average attack: Rate items with average rating
- - Segment attack: Rate items in a specific segment
- Selected items: popular items or items common features
- Filler items: randomly chosen
- Unrated items: not rated by user
- Target item: item of interest

- Random attack: 
- - Selected items not considered
- - Filler items rated at random based on overall average and standard deviation
- Average attack:
- - Selected items not considered
- - Filler items rated by generating values based on item's known average rating and standard 
    deviation.
- Bandwagon attack:
- - Assign high ratings for selected items
- - Filler items chosen randomly
- - Don't need to know average of item ratings
- Segment attack:
- - Starts from particular item
- - Looks at other similar items and identify user community
- - Selected similar items then assigned high ratings
- - Targets the right community
- - Random values to filler items

## Recommender Systems: Attack Detection
- Degree of Similarity with Top Neighbours
- Rating Deviation from Mean Agreement

## Recommender System Software
- Taken from [here](https://github.com/grahamjenson/recommender-systems)
- Software as a Service (SaaS) Recommendation Systems
- Open Source Recommendation Systems - Surprise, LightFM, Crab, LensKit, MyMediaLite, LibRec, Mahout, LensKit, PredictionIO, RecLab, RecommenderLab, EasyRec, Apache PredictionIO, Apache Mahout, Apache LensKit
- Non-SaaS Recommender Systems - Turi Create
- Academic Recommender Systems - LibRec, RankSys, LIBMF
- Benchmarking Recommender Systems - TagRec, RiVal
- Media Recommendation Applications - Jinni, Gyde, Pandora

## Challenges in Recommendation Systems
- Missing ratings
- Recency of rating
- User specific models
- Profit driven recommendation systems
- Explainable recommendation systems
- Properly evaluate new methods
- Active learning
- Privacy
