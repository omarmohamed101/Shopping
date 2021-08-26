import csv
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    df = pd.read_csv('shopping.csv')
    features_names = df.columns[:-1]
    label_name = df.columns[-1]

    df['Weekend'].replace({False: 0, True: 1}, inplace=True)
    df['VisitorType'].replace({'New_Visitor': 0,'Other': 0, 'Returning_Visitor': 1}, inplace=True)
    df['Month'].replace({'Feb': 1, 'Mar': 2, 'May': 4, 'Oct': 9, 'June': 5, 'Jul': 6, 'Aug': 7, 'Nov': 10, 'Sep': 8,
       'Dec': 11}, inplace=True)
    df['Revenue'].replace({False: 0, True: 1}, inplace=True)

    evidence = df[features_names]
    labels = df[label_name]

    evidence = evidence.to_numpy().tolist()
    labels = labels.to_numpy().tolist()

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    labels = np.array(labels)
    predictions = np.array(predictions)

    true_positive = sum(labels * predictions)
    true_negative = sum((predictions + labels) == 0)
    false_positive = sum((predictions == 1) & (labels == 0))
    false_negative = sum((predictions == 0) & (labels == 1))

    sensitivity = float(true_positive) / (true_positive + false_negative)
    specificty = float(true_negative) / (true_negative + false_positive)

    return (sensitivity, specificty)    



if __name__ == "__main__":
    main()
