# minimax-risk-classifier
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/) 
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)<br/>
Supervised classification techniques use training samples to find classification rules with small expected 0-1 loss. Conventional methods achieve efficient learn- ing and out-of-sample generalization by minimizing surrogate losses over specific families of rules. This paper presents minimax risk classifiers (MRCs) that do not rely on a choice of surrogate loss and family of rules. MRCs achieve efficient learning and out-of-sample generalization by minimizing worst-case expected 0-1 loss w.r.t. uncertainty sets that are defined by linear constraints and include the true underlying distribution. In addition, MRCs’ learning stage provides perfor- mance guarantees as lower and upper tight bounds for expected 0-1 loss. We also present MRCs’ finite-sample generalization bounds in terms of training size and smallest minimax risk, and show their competitive classification performance w.r.t. state-of-the-art techniques using benchmark datasets.
![Test Image 1](docs/images/3DTest.png)
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
        <td>.21</td>
        <td>.20 ± .04</td>
        <td>.24 ± .04</td>
        <td>.22 ± .04</td>
        <td>.18 ± .03</td>
        <td>.21 ± .06</td>
        <td>.18 ± .03</td>
        <td>.22 ± .04</td>
    </tr>
    <tr>
        <td>Haberman</td>
        <td>.24</td>
        <td>.27 ± .03</td>
        <td>.27</td>
        <td>.24 ± .03</td>
        <td>.39 ± .14</td>
        <td>.30 ± .07</td>
        <td>.26 ± .04</td>
        <td>.35 ± .12</td>
        <td>.25 ± .04</td>
        <td>.27 ± .02</td>
    </tr>
    <tr>
        <td>Indian liv.</td>
        <td>.28</td>
        <td>.29 ± .01</td>
        <td>.30</td>
        <td>.44 ± .08</td>
        <td>.35 ± .09</td>
        <td>.34 ± .05</td>
        <td>.29 ± .02</td>
        <td>.30 ± .05</td>
        <td>.29 ± .01</td>
        <td>.29 ± .01</td>
    </tr>
    <tr>
        <td>Diabetes</td>
        <td>.22</td>
        <td>.26 ± .03</td>
        <td>.28</td>
        <td>.26 ± .03</td>
        <td>.29 ± .07</td>
        <td>.26 ± .05</td>
        <td>.24 ± .04</td>
        <td>.26 ± .05</td>
        <td>.24 ± .04</td>
        <td>.34 ± .04</td>
    </tr>
    <tr>
        <td>Credit</td>
        <td>.12</td>
        <td>.15 ± .18</td>
        <td>.17</td>
        <td>.22 ± .07</td>
        <td>.22 ± .14</td>
        <td>.14 ± .09</td>
        <td>.16 ± .17</td>
        <td>.17 ± .15</td>
        <td>.15 ± .18</td>
        <td>.14 ± .04</td>
    </tr>
    <tr>
        <td>Glass</td>
        <td>.22</td>
        <td>.36 ± .08</td>
        <td>.47</td>
        <td>.64 ± .04</td>
        <td>.39 ± .18</td>
        <td>.34 ± .08</td>
        <td>.34 ± .11</td>
        <td>.40 ± .14</td>
        <td>.42 ± .14</td>
        <td>.35 ± .08</td>
    </tr>
    <tr>
        <td>Avg. rank</td>
        <td colspan="3">2.7</td>
        <td>5.1</td>
        <td>7.0</td>
        <td>3.8</td>
        <td>2.0</td>
        <td>5.3</td>
        <td>2.5</td>
        <td>3.8</td>
    </tr>
</table>


# Contributing
