You might perform feature engineering to:

- improve a model's predictive performance
- reduce computational or data needs
- improve interpretability of the results

Linear models, for instance, are only able to learn linear relationships.
So, when using a linear model, your goal is to transform the features to make their relationship to the target linear.

![LinearRegressionGraph](https://storage.googleapis.com/kaggle-media/learn/images/5D1z24N.png)

`A linear model fits poorly with only Length as feature.`



![](https://storage.googleapis.com/kaggle-media/learn/images/BLRsYOK.png)

`Left: The fit to Area is much better. Right: Which makes the fit to Length better as well.`

If we square the Length feature to get 'Area', however, we create a linear relationship. Adding Area to the feature set means this linear model can now fit a parabola. Squaring a feature, in other words, gave the linear model the ability to fit squared features.



## Mutual Information

First encountering a new dataset can sometimes feel overwhelming. You might be presented with hundreds or thousands of features without even a description to go by. Where do you even begin?

The metric we'll use is called "mutual information". Mutual information is a lot like correlation in that it measures a relationship between two quantities. The advantage of mutual information is that it can detect any kind of relationship, while correlation only detects linear relationships.

Mutual information is a great general-purpose metric and especially useful at the start of feature development when you might not know what model you'd like to use yet. It is:

- easy to use and interpret
- computationally efficient
- theoretically well-founded
- resistant to overfitting
- able to detect any kind of relationship


## Creating Features


#### Mathematical Transforms

> Relationships among numerical features are often expressed through mathematical formulas, which you'll frequently come across as part of your domain research. <br/>
>In Pandas, you can apply arithmetic operations to columns just as if they were ordinary numbers.

Tips on Discovering New Features

1. Understand the features. Refer to your dataset's data documentation, if available.
2. Research the problem domain to acquire domain knowledge. If your problem is predicting house prices, do some research on real-estate for instance. Wikipedia can be a good starting point, but books and journal articles will often have the best information.
3. Study previous work. [Solution write-ups](https://www.kaggle.com/code/sudalairajkumar/winning-solutions-of-kaggle-competitions) from past Kaggle competitions are a great resource.
4. Use data visualization. Visualization can reveal pathologies in the distribution of a feature or complicated relationships that could be simplified. Be sure to visualize your dataset as you work through the feature engineering process.

#### Counts

Features describing the presence or absence of something often come in sets, the set of risk factors for a disease, say. You can aggregate such features by creating a count.
These features will be binary (1 for Present, 0 for Absent) or boolean (True or False). In Python, booleans can be added up just as if they were integers.
In Traffic Accidents are several features indicating whether some roadway object was near the accident. This will create a count of the total number of roadway features nearby using the sum method:


### Building-Up and Breaking-Down Features

Often you'll have complex strings that can usefully be broken into simpler pieces. Some common examples:

- ID numbers: '123-45-6789'
- Phone numbers: '(999) 555-0123'
- Street addresses: '8241 Kaggle Ln., Goose City, NV'
- Internet addresses: 'http://www.kaggle.com
- Product codes: '0 36000 29145 2'
- Dates and times: 'Mon Sep 30 07:06:05 2013'


##### Elsewhere on Kaggle Learn


- For dates and times, see [Parsing Dates](https://www.kaggle.com/alexisbcook/parsing-dates) from our Data Cleaning course.
- For latitudes and longitudes, see our [Geospatial Analysis course](https://www.kaggle.com/learn/geospatial-analysis).


### Group Transforms

Finally we have Group transforms, which aggregate information across multiple rows grouped by some category. With a group transform you can create features like: "the average income of a person's state of residence," or "the proportion of movies released on a weekday, by genre." If you had discovered a category interaction, a group transform over that categry could be something good to investigate.

Using an aggregation function, a group transform combines two features: a categorical feature that provides the grouping and another feature whose values you wish to aggregate. For an "average income by state", you would choose State for the grouping feature, mean for the aggregation function, and Income for the aggregated feature. To compute this in Pandas, we use the groupby and transform methods:


#### Tips on Creating Features

It's good to keep in mind your model's own strengths and weaknesses when creating features. Here are some guidelines:
- **Linear models** learn sums and differences naturally, but can't learn anything more complex.
- Ratios seem to be difficult for most models to learn. Ratio combinations often lead to some easy performance gains.
- **Linear models** and neural nets generally do better with normalized features. Neural nets especially need features scaled to values not too far from 0. Tree-based models (like random forests and XGBoost) can sometimes benefit from normalization, but usually much less so.
- **Tree models** can learn to approximate almost any combination of features, but when a combination is especially important they can still benefit from having it explicitly created, especially when data is limited.
- Counts are especially helpful for **tree models**, since these models don't have a natural way of aggregating information across many features at once.


### Clustering With K-Means

> Cluster Labels as a Feature: <br/>
> Applied to a single real-valued feature, clustering acts like a traditional "binning" or "discretization" transform. <br/>
> On multiple features, it's like "multi-dimensional binning" (sometimes called vector quantization)


![clustering](https://storage.googleapis.com/kaggle-media/learn/images/sr3pdYI.png)


It's important to remember that this Cluster feature is categorical. Here, it's shown with a label encoding (that is, as a sequence of integers) as a typical clustering algorithm would produce; depending on your model, a one-hot encoding may be more appropriate.

The motivating idea for adding cluster labels is that the clusters will break up complicated relationships across features into simpler chunks. Our model can then just learn the simpler chunks one-by-one instead having to learn the complicated whole all at once. It's a "divide and conquer" strategy.

![clustering_2](https://storage.googleapis.com/kaggle-media/learn/images/rraXFed.png)