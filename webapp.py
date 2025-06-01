# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#app title
st.title("""
         Explore Different Machine Learning Models
         Dekhny k liye niche se model select kren
         """)
# Load the iris dataset
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Wine", "Breast Cancer"))

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

# define a function to load the dataset
def load_dataset(name):
    data = None
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()

    x=data.data
    y=data.target
    return x,y

# ab function ko call kren gy
x, y = load_dataset(dataset_name)

# dataset ki shape
st.write("Shape of dataset:", x.shape)
st.write("Number of classes:", len(np.unique(y)))

# add different classifier parameters in user interface
         
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C # its the degree of correct classification
    elif clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "Random Forest":
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        max_depth = st.sidebar.slider("max_depth", 1, 20)
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        params["max_depth"] = max_depth
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["n_estimators"] = n_estimators
    return params

# ab function ko call kren gy
params = add_parameter_ui(classifier_name)

# define a function to get the classifier
def get_classifier(clf_name, params):
    clf = None
    if clf_name == "SVM":
        clf = SVC(C=params["C"])
    elif clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=1234)
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=1234)
    return clf


if st.checkbox("Show code"):
    with st.echo():
     clf = get_classifier(classifier_name, params)

# split the dataset into training and testing sets
     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

        # fit the classifier on the training data
     clf.fit(x_train, y_train)

        # make predictions on the test data
     y_pred = clf.predict(x_test)

    # calculate the accuracy of the classifier
     accuracy = accuracy_score(y_test, y_pred)
#ab function ko call kren gy
clf = get_classifier(classifier_name, params)

        # split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

        # fit the classifier on the training data
clf.fit(x_train, y_train)

        # make predictions on the test data
y_pred = clf.predict(x_test)

    # calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

# display the accuracy
st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {accuracy:.2f}")

# plot the dataset
pca= PCA(2)
X_projected = pca.fit_transform(x)
# create a dataframe for the PCA data
x1=X_projected[:, 0]
x2=X_projected[:, 1]

fig= plt.figure()
plt.scatter(x1, x2, c=y,alpha=0.8, cmap='viridis')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

# display the plot in the Streamlit app
st.pyplot(fig)


