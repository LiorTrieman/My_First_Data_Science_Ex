"""
PP Home Assignment
Lior Trieman
"""

# ------------------ #
# *1* IMPORT MODULES #
# ------------------ #
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import metrics


def impute_data_median(data_frame, vector_ind):
    vector = data_frame.iloc[:, vector_ind]
    imp = SimpleImputer(strategy="median")
    vector_df = pd.DataFrame(vector)
    vector_no_nan = imp.fit_transform(vector_df)
    vector_no_nan = vector_no_nan.tolist()
    vector_no_nan_list = sum(vector_no_nan, [])
    return vector_no_nan_list


def impute_df(df):
    imputed_df = ({'#viewed_ads': impute_data_median(df, 0), '#times_visited_website': impute_data_median
    (df, 1), 'age': impute_data_median(df, 8)})
    df.update(imputed_df)  # update matrix with imputed data
    return df


def drop_features(df):  # dropping "timestamp" - not relevant?
    return df.drop(['timestamp'], axis=1)


def all_lower_case_df(df):
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)  # converting all list to \
    return df


def all_float_df(df):
    df = df.applymap(lambda s: float(s) if type(s) == int else s)  # converting all numeric \
    return df


def transform_features(df):
    df = all_lower_case_df(df)
    df = all_float_df(df)
    df = drop_features(df)
    return df

def encode_features(df_train, df_test):
    features = ['target_product_price_color', 'target_product_category', 'shopper_segment', 'delivery_time']
    df_combined = pd.concat(
        [df_train[features], df_test[features]])  # catch categories that might be in test set also

    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test


if __name__ == "__main__":

    # data file and test file
    DATA_FILE = "interview_dataset_train"  # name of file with data to read
    TEST_FILE = "interview_dataset_test_no_tags"  # name of file with  test data (for submitting the output of my model)

    # -------------------- #
    # *2* LOADING THE DATA #
    # -------------------- #
    data_train = pd.read_csv(DATA_FILE, sep="\t", header=0)
    data_test = pd.read_csv(TEST_FILE, sep="\t", header=0)

    print(data_train.sample(3))  # look at the data

    # ------------------------ #
    # *3* GET TO KNOW THE DATA #
    # ------------------------ #

    plt.figure()
    sns.countplot(x="tag", data=data_train)  # most of the tags are '0'
    plt.figure()
    sns.countplot(x="target_product_price_color", data=data_train)  # most of the colors are "red"
    plt.figure()
    sns.countplot(x="tag", hue="shopper_segment", data=data_train)
    plt.figure()
    data_train["age"].plot.hist()
    plt.xlabel('Age')
    plt.show()
    print("tags distribution:", data_train['tag'].value_counts())

    # SOME INITIAL TRAIN/TEST DATA STATISTICS:

    print("age-data_train", data_train.age.describe())
    print("age-data_test", data_test.age.describe())
    print("target_product_price-data_train", data_train.target_product_price.describe())
    print("target_product_price-data_test", data_test.target_product_price.describe())
    print("shopper_segment-data_train", data_train.shopper_segment.describe())
    print("shopper_segment-data_test", data_test.shopper_segment.describe())
    print("target_product_price_color-data_train", data_train.target_product_price_color.describe())
    print("target_product_price_color-data_test", data_test.target_product_price_color.describe())
    print("delivery_time-data_train", data_train.delivery_time.describe())
    print("delivery_time-data_test", data_test.delivery_time.describe())
    print("target_product_description_length-data_train", data_train.target_product_description_length.describe())
    print("target_product_description_length-data_test", data_test.target_product_description_length.describe())

    # ------------------- #
    # *4* DATA IMPUTATION #
    # ------------------- #

    impute_df(data_test)
    impute_df(data_train)

    # ------------------------- #
    # *5* TRANSFORMING FEATURES #
    # ------------------------- #

    # apply on data:
    data_train = transform_features(data_train)
    data_test = transform_features(data_test)
    data_train.head()

    print("num of nulls in data:", data_train.isna().sum())  # to see if we have nulls in matrix
    plt.figure()
    sns.heatmap(data_train.isnull())
    plt.show()

    # ------------------------------------------------------ #
    # *6* FINAL PRE-PROCESSING (for both train and test data)#
    # ------------------------------------------------------ #

    data_train, data_test = encode_features(data_train, data_test)
    data_train.head()
    data_test.head()

    # ---------------------------------- #
    # *7* SPLITTING UP THE TRAINING DATA #
    # ---------------------------------- #

    X_all = data_train.drop(['tag'], axis=1)
    y_all = data_train['tag']

    NUM_TEST = 0.25  # training 75% of the data, then testing against the other 25%.

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=NUM_TEST, random_state=42)

    # --------------------------------- #
    # *8* FITTING & TUNING AN ALGORITHM #
    # --------------------------------- #

    # optional models:
    # ----------------
    #   RandomForestClassifier
    #   Support Vector Machines (SVM)
    #   LogisticRegression

    clf = RandomForestClassifier()

    parameters = {'n_estimators': [4, 6, 9],
                  'max_features': ['log2', 'sqrt','auto'],
                  'criterion': ['entropy', 'gini'],
                  'max_depth': [2, 3, 5, 10],
                  'min_samples_split': [2, 3, 5],
                  'min_samples_leaf': [1, 5, 8]
                  }

    warnings.filterwarnings('ignore')
    # Run the grid search
    grid_obj = GridSearchCV(clf, parameters, scoring='f1', cv=5)
    grid_obj = grid_obj.fit(X_train, y_train)

    # Set the clf to the best combination of parameters
    clf = grid_obj.best_estimator_
    print("start calculating the model...")
    # Fit the best algorithm to the data.
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    # Accuracy = TP+TN/TP+FP+FN+TN
    # Precision = TP/TP+FP
    # Recall = TP/TP+FN
    # F1 Score = 2*(Recall * Precision) / (Recall + Precision) - BEST FIT FOR UNBALANCED TAGS
    print("classification_report", metrics.classification_report(y_test, predictions))
    print("confusion_matrix", metrics.confusion_matrix(y_test, predictions))
    accuracy_score(y_test, predictions)
    print("predicted:", predictions)
    print("y_test", y_test)
    print("Accuracy:", metrics.accuracy_score(y_test, predictions))
    print("Precision:", metrics.precision_score(y_test, predictions))
    print("Recall:", metrics.recall_score(y_test, predictions))
    precision = metrics.precision_score(y_test, predictions)
    recall = metrics.recall_score(y_test, predictions)
    F1 = 2 * (precision * recall) / (precision + recall)
    print("F1 score = ", F1)

    # -------------------------------- #
    # *9* Predict the Actual Test Data #
    # -------------------------------- #
    print("data_test processed:", data_test.head(11))
    print("data_train processed:", data_train.head(11))
    predictions = clf.predict(data_test)  # predict on test data
    # data_test.to_csv("Test_data_postprocess.csv", index=False)
    submission = pd.DataFrame({'Tags': predictions})
    submission.to_csv("Tags_New.csv", index=False)

    plt.figure()
    sns.countplot(x="Tags", data=submission)  # most of the tags are '0'
    plt.show()
