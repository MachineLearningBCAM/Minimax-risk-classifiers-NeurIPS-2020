from os.path import join, dirname

import csv
import numpy as np
from sklearn.datasets.base import Bunch, load_data


def load_adult(return_X_y=False):
    """Load and return the adult incomes prediction dataset (classification).

    =================   ==============
    Classes                          2
    Samples per class    [37155,11687]
    Samples total                48882
    Dimensionality                  14
    Features             int, positive
    =================   ==============

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'adult.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'adult.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.int)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)


def load_diabetes(return_X_y=False):
    """Load and return the Pima Indians Diabetes dataset (classification).

    =================   =====================
    Classes                                 2
    Samples per class               [500,268]
    Samples total                         668
    Dimensionality                          8
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'diabetes.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'diabetes.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.int)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)


def load_credit(return_X_y=False):
    """Load and return the Credit Approval prediction dataset (classification).

    =================   =====================
    Classes                                 2
    Samples per class               383, 307]
    Samples total                         690
    Dimensionality                         15
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'credit.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'credit.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int64)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            try:
                data[i] = np.asarray(d[:-1], dtype=np.float64)
            except ValueError:
                print(i, d[:-1])
            target[i] = np.asarray(d[-1], dtype=np.int64)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)


def load_magic(return_X_y=False):
    """Load and return the Magic Gamma Telescope dataset (classification).

    =========================================
    Classes                                 2
    Samples per class            [6688,12332]
    Samples total                       19020
    Dimensionality                         10
    Features                            float
    =========================================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'magic.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'magic.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.str)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.str)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)


def load_glass(return_X_y=False):
    """Load and return the Glass Identification Data Set (classification).

    ===========================================
    Classes                                   6
    Samples per class    [70, 76, 17, 29, 13, 9]
    Samples total                           214
    Dimensionality                            9
    Features                              float
    ===========================================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of glass csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'glass.csv')
    with open(join(module_path, 'descr', 'glass.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['RI: refractive index',
                                "Na: Sodium (unit measurement: "
                                "weight percent in corresponding oxide, "
                                "as are attributes 4-10)",
                                'Mg: Magnesium ',
                                'Al: Aluminim',
                                'Si: Silicon',
                                'K: Potassium',
                                'Ca: Calcium',
                                'Ba: Barium',
                                'Fe: Iron'])


def load_haberman(return_X_y=False):
    """Load and return the Haberman's Survival Data Set (classification).

    ==============================
    Classes                      2
    Samples per class    [225, 82]
    Samples total              306
    Dimensionality               3
    Features                   int
    ==============================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of haberman csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'haberman.csv')
    with open(join(module_path, 'descr', 'haberman.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['PatientAge',
                                'OperationYear',
                                'PositiveAxillaryNodesDetected'])


def load_mammographic(return_X_y=False):
    """Load and return the Mammographic Mass Data Set (classification).

    ==============================
    Classes                      2
    Samples per class    [516, 445]
    Samples total              961
    Dimensionality               5
    Features                   int
    ==============================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of mammographic csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'mammographic.csv')
    with open(join(module_path, 'descr', 'mammographic.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['BI-RADS',
                                'age',
                                'shape',
                                'margin',
                                'density'])


def load_indian_liver(return_X_y=False):
    """Load and return the Indian Liver Patient Data Set
    (classification).

    =========================================================
    Classes                                                 2
    Samples per class                              [416, 167]
    Samples total                                         583
    Dimensionality                                         10
    Features                                       int, float
    Missing Values                                     4 (nan)
    =========================================================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of satellite csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    """data, target, target_names = load_data(module_path,
                                           'indianLiverPatient.csv')"""
    with open(join(module_path, 'data',
                   'indianLiverPatient.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=np.int)
    with open(join(module_path, 'descr',
                   'indianLiverPatient.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['Age of the patient',
                                'Gender of the patient',
                                'Total Bilirubin',
                                'Direct Bilirubin',
                                'Alkaline Phosphotase',
                                'Alamine Aminotransferase',
                                'Aspartate Aminotransferase',
                                'Total Protiens',
                                'Albumin',
                                'A/G Ratio'])