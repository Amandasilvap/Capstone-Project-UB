# Capstone-Project-UB
Final Project of University of Barcelona


## Team Members
- Alejandro Chaqués  - https://github.com/janoxakes
- Amanda da Silva Pinto - https://github.com/Amandasilvap
- Pablo Cebriá Cortina - https://github.com/pabcebco

## Project
We have one Datasets.zip with all datasets and another one **Code** with the final Notebook

### Sumary

This is the closing project of the Postgraduate Introduction to Machine Learning and Data Science course. 

We used the 2022 Recsys Challenge, focuses on fashion recommendations, as the basis of the project. When given user sessions, purchase data and content data about items, can you accurately predict which fashion item will be bought at the end of the session?

### Dataset
As part of this challenge, Dressipi will be releasing a public dataset of 1.1 million online retail sessions that resulted in a purchase. In addition, all items in the dataset have been labeled with content data and the labels are supplied. We refer to the label data as item features. The dataset is sampled and anonymized.

- Sessions: The items that were viewed in a session. In this dataset a session is equal to a day, so a session is one user's activity on one day.
- Purchases: The purchase that happened at the end of the session. One purchased item per session.
- Item features: The label data of items. Things like “color: green,” “neckline: v-neck,” etc.

## References

- http://www.recsyschallenge.com/2022/
- https://www.dressipi-recsys2022.com/
- https://www.dressipi-recsys2022.com/profile/download_dataset

## About the code
The Recsys 2022 Challenge aims to predict the probability that a user will buy one of the products on a pre-selected list. To do this, 1 million sessions are provided that illustrate the journey of each one through the different articles and, ultimately, the product purchased.

The total number of items present in the item_features.csv dataset amounts to 23,691 (although only 23,496 have been displayed in the train_sessions.csv dataset), of which only 4,990 are candidates for prediction.

In view of the evaluation metric used in the challenge, it is established that the objective of this project is, given a session, to order from highest to lowest the 100 candidate items with the highest probability of being acquired at the end of the session.

#### Interesting Question We Can Answer with Data

```markdown
What are the 100 candidate products most likely to be purchased in each session based on the products visited 
and their characteristics?

```

### Data Acquisition

Due to the magnitude of the problem and the computational limitation of the resources available during the execution of this challenge, it has been decided to limit the instances available in the dataset to those that have a product from the list of candidates as their purchased product. That is, all those instances that have as purchased item one that appears in the candidate_items.csv dataset will be removed from the train dataset.

Another of the methods that we evaluated was to give a prediction of the 23,691 items in item_features.csv and, for the evaluation, consider only the 4,990 in candidate_items.csv. However, due to hardware limits of the equipment used in this challenge, we had to rule out this option.

**The logic used, knowing that it is incomplete, assumes that all those sessions that do not end in a candidate article do not provide information to the machine learning model and can be discarded.**

### Feature Engineering

It should be noted that, despite the fact that the notebook includes multiple feature engineering elements, for the final modeling it has been decided to leave all these elements out of the train. This decision has been made based on the limitations established at the beginning regarding the hardware necessary to process the available datasets.

For more details see GitHub Repository [Capstone Project](https://github.com/Amandasilvap/Capstone-Project-UB/blob/main/Code/Capstone_final.ipynb) or our [GitHub Page] (https://amandasilvap.github.io/Capstone-Project-UB/).

### Conclusions

Due to the multiple decisions we have made throughout this project, there is no way to compare this result with those obtained in the RecSys challenge. Therefore, we have decided to make another prediction with a simpler linear model and compare the results.

In this particular case, and considering max_depth = 3, the result obtained is 0.02385566748579892 compared to 0.002111074682055869 obtained with the linear model, assuming an improvement of approximately 11 times.

Taking into account that there are 4,990 target articles, and assigning them randomly would have a probability of success of 0.0002 (without considering the case evaluation method), we consider that the defined model substantially improves the random model and simpler models such as the linear one.
