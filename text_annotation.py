import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a synthetic dataset
x, y  = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Split the dataset into intital training set and pool set
x_train, x_pool, y_train, y_pool = train_test_split(x, y, test_size=0.9, random_state=42)

# Initialize the active learning loop
iterations = 10
batch_size = 10
model = LogisticRegression(random_state=42)

for i in range(iterations):
    print("Iteration {}:".format(i+1))

    # Train the model on the current training set
    model.fit(x_train, y_train)

    # Predict the labels of the unlabeled instances in the pool set
    y_pool_pred = model.predict(x_pool)

    ### below
    y_pool_prob = model.predict_proba(x_pool)
    # entropy = -np.sum(y_pool_prob * np.log(y_pool_prob), axis=1)
    # query_idx = np.argsort(entropy)[-batch_size:]
    least_confidence = [np.max(y_pool_prob) - np.average(prob_val) for prob_val in y_pool_prob]
    query_idx = np.argsort(least_confidence)[-batch_size:]
    ### above

    x_query = x_pool[query_idx]
    y_query = y_pool[query_idx]

    # Add the labeled instances to the training set and remove them from the pool set
    x_train = np.concatenate([x_train, x_query])
    y_train = np.concatenate([y_train, y_query])
    x_pool = np.delete(x_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)

    # Compute and print the accuracy of the model on the test set
    y_test_pred = model.predict(x_pool)
    accuracy = accuracy_score(y_pool, y_test_pred)
    print("Accuracy: {:3f}\n".format(accuracy))
