# minimax-risk-classifier
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/) 
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)<br/>
Supervised classification techniques use training samples to find classification rules with small expected 0-1 loss. Conventional methods achieve efficient learn- ing and out-of-sample generalization by minimizing surrogate losses over specific families of rules. This paper presents minimax risk classifiers (MRCs) that do not rely on a choice of surrogate loss and family of rules. MRCs achieve efficient learning and out-of-sample generalization by minimizing worst-case expected 0-1 loss w.r.t. uncertainty sets that are defined by linear constraints and include the true underlying distribution. In addition, MRCs’ learning stage provides perfor- mance guarantees as lower and upper tight bounds for expected 0-1 loss. We also present MRCs’ finite-sample generalization bounds in terms of training size and smallest minimax risk, and show their competitive classification performance w.r.t. state-of-the-art techniques using benchmark datasets.
# Requirements
[![Generic badge](https://img.shields.io/badge/Python-2.X|3.X-blue.svg)](https://shields.io/)<br/>
We will need have installed the following libraries:
* numpy
* sklearn
* cvxpy

# Training
To create an instance of the MRC classifier we must first define the following parameters:
* r: number of different classes in the prediction
* phi: Features of the MRC

For the LPC instance we also need to know:
* m = number of product thresholds
* k = maximum number of univariate thresholds for each dimension
* d = number of predictor variables / number of columns of dataset X

MRC_model = MRC(r=r, phi=PhiThreshold(r=r, m=int(mumVars/r), k=int(mumVars/(r*d))), s=s)<br/>

To train our MRC classifier, we have to pass as parameters, the training data set (X_train) with the predictor variables and the label of each case (y_train) <br/>
MRC_model.fit(X_train, y_train)<br/>

# Evaluation
To collect the results predicted by our classifier, we run the following line:
y_pred= MRC_model.predict(X_test)


# Results
<table>
    <tr>
        <th>Data set</th>
        <th>LB</th>
        <th>MRC</th>
        <th>UB</th>
        <th>QDA</th>
        <th>DT</th>
        <th>KNN</th>
        <th>SVM</th>
        <th>RF</th>
        <th>AMC</th>
        <th>MEM</th>
    </tr>
    <tr>
        <td>Mammog.</td>
        <td>.16</td>
        <td>.18 ± .04</td>
    </tr>
</table>


# Contributing
