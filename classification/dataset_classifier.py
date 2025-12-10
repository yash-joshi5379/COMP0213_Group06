from classification.ClassifierModel import Classifier


def classify_dataset(model, dataset):
    """
    Classification pipeline: train, test, and save a classifier model on a given dataset.

    it creates a classifier, splits the dataset, trains the model with hyperparameter
    optimization, evaluates it on the test set, and saves the trained model to disk.

    Args:
        model (str): The type of model to train ('R' for Random Forest, 'L' for
            Logistic Regression, 'N' for Neural Network, or 'S' for SVM).
        dataset (str): Path to the CSV dataset file containing features and labels.
    """
    user_classifier = Classifier(dataset, model)
    user_classifier.split_dataset()

    print("Training model...\n")
    user_classifier.train_model()

    print("Testing model...\n")
    user_classifier.test_model()

    print("Saving model...\n")
    user_classifier.save_model()
