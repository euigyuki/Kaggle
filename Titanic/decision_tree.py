import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer

def preprocess_data(data, drop_cols):
    """
    Preprocess the data by filling missing values, dropping unnecessary columns,
    and converting categorical variables to numerical.
    """
    # Fill missing values in the 'Age' column with the median age
    median_age = data['Age'].median()
    data['Age'] = data['Age'].fillna(median_age)

    # Drop unnecessary columns
    data.drop(drop_cols, axis=1, inplace=True)

    # Convert categorical variables to numerical
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'])

    return data

def train_model(X_train, y_train, X_test=None, y_test=None):
    """
    Train a decision tree classifier on the training data.
    """
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, X_test=None, y_test=None):
    """
    Evaluate the model's performance on the training and test data.
    """
    # Evaluate on training data
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Accuracy on training data: {train_accuracy:.4f}")

    # Evaluate on test data if provided
    if X_test is not None and y_test is not None:
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Accuracy on test data: {test_accuracy:.4f}")

def create_submission(test_data, y_pred):
    """
    Create a submission DataFrame with the predicted values.
    """
    submission = pd.DataFrame({
        'PassengerId': test_data['PassengerId'],
        'Survived': y_pred
    })
    submission.to_csv('decision_tree_submission.csv', index=False)

def main():
    # Load the  dataset
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')


    # Preprocess the  data
    train_data = preprocess_data(train_data, ['PassengerId', 'Name', 'Ticket', 'Cabin'])
    test_data = preprocess_data(test_data, ['Name', 'Ticket', 'Cabin'])

    # Split the data into features and target
    X_train = train_data.drop('Survived', axis=1)
    y_train = train_data['Survived']
    X_test = test_data.drop('PassengerId', axis=1)


    # Train the model
    model = train_model(X_train, y_train, X_test=X_test)

    # Evaluate the model on the training data
    evaluate_model(model, X_train, y_train)



    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Create the submission file
    create_submission(test_data, y_pred)

if __name__ == '__main__':
    main()