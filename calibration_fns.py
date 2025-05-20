# citation: https://github.com/avalanchesiqi/pyquantifier/blob/main/pyquantifier/calibration_curve.py
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

import os
import numpy as np
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV


def get_bin_idx(score, size=10):
    return min(int(score * size), size-1)

def prepare_canvas():
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major')
    return ax

class CalibrationCurve:
    """
    Implementation of a calibration curve.
    """
    def __init__(self):
        self.num_bin = 100
        self.x_axis = np.arange(0.05/self.num_bin, 1, 1/self.num_bin)
        self.y_axis = self.get_calibrated_probs(self.x_axis)

    def get_calibrated_prob(self, cx):
        pass

    def get_calibrated_probs(self, cxs):
        return np.array([self.get_calibrated_prob(cx) for cx in cxs])

    def plot(self, **kwds):
        ax = kwds.pop('ax', None)
        if ax is None:
            ax = prepare_canvas()

        show_diagonal = kwds.pop('show_diagonal', False)
        filled = kwds.pop('filled', True)
        lc = kwds.pop('lc', 'k')
        label = kwds.pop('label', None)

        bin_width = 1 / self.num_bin
        bin_margin = bin_width / 2

        for bin_idx in range(self.num_bin):
            x = self.x_axis[bin_idx]
            y = self.y_axis[bin_idx]
            left_coord = x - bin_margin
            right_coord = x + bin_margin

            ax.plot([left_coord, right_coord], [y, y], c=lc, lw=2, zorder=40)
            if bin_idx > 0:
                # left edge of the bin
                prev_y = self.y_axis[bin_idx - 1]
                ax.plot([left_coord, left_coord], [prev_y, y], c=lc, lw=2, zorder=40)
            # next_y = self.y_axis[bin_idx + 1] if bin_idx < num_bin - 1 else 1
            # ax.plot([right_coord, right_coord], [y, next_y], c='k', lw=2, zorder=40)

            if filled:
                ax.fill_between([left_coord, right_coord],
                                [y, y],
                                [0, 0],
                                alpha=x, lw=0)

                ax.fill_between([left_coord, right_coord],
                                [1, 1],
                                [y, y],
                                alpha=x, lw=0)

        if show_diagonal:
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
        
        if label is not None:
            ax.legend([label], loc='upper left')

        ax.set_xlabel('$C(x)$')
        ax.set_ylabel('$P(y=1|C(x))$')
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)



class PlattScaling(CalibrationCurve):
    """
    A logistic calibration curve
    """
    def __init__(self, model):
        self.model = model
        super().__init__()
    
    def get_calibrated_prob(self, cx):
        return self.model.predict_proba(cx.reshape(1, -1))[0, 1]

    def get_calibrated_probs(self, cxs):
        return self.model.predict_proba(cxs.reshape(-1, 1))[:, 1]


class BinnedCalibrationCurve(CalibrationCurve):
    def __init__(self, x_axis, y_axis):
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.num_bin = len(self.x_axis)

    def get_calibrated_prob(self, cx):
        return self.y_axis[get_bin_idx(cx, size=self.num_bin)]


def generate_calibration_curve_platt(df, code = "L", other_codes = ["O","P","S"]):
    cx_col_name = "probs_" + code
    train_CX = df[cx_col_name].values.reshape(-1, 1)
    train_GT = df['gt_label'].map({other_codes[0]: 0, other_codes[1]: 0, other_codes[2]: 0, code: 1}).values
    
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=547)

    # use cross-validation to find the best c value
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    logreg_model = LogisticRegression(solver='lbfgs', fit_intercept=True)

    grid_search = GridSearchCV(logreg_model, param_grid, cv=kf, scoring='accuracy')
    grid_search.fit(train_CX, train_GT)

    # Get the best model
    best_model = grid_search.best_estimator_
    return PlattScaling(model=best_model)

def generate_calibration_curve_binned(df, num_bin = 10, code = "L", other_codes = ["O","P","S"]):
    cx_col_name = "probs_" + code
    x_axis = np.arange(0.5/num_bin, 1, 1/num_bin)
    df['gt_label'] = df['gt_label'].map({other_codes[0]: 0, other_codes[1]: 0, other_codes[2]: 0, code: 1}).values
    # create a new column based on the probs column
    df['bin'] = df.apply(lambda row: get_bin_idx(row[cx_col_name], size=num_bin), axis=1)
    # Calculate fraction postitive for each bin
    y_axis = [0 if len(df[df['bin'] == bin_idx])==0 else (len(df[(df['bin'] == bin_idx) & (df['gt_label'] == 1)]) / len(df[df['bin'] == bin_idx]) ) 
              for bin_idx, _ in enumerate(x_axis)]
    return BinnedCalibrationCurve(x_axis=x_axis, y_axis=y_axis)

def extrinsic_estimate(df, calibration_curve: CalibrationCurve, code = "L"):
    cx_col_name = "probs_" + code
    df['calibr_pos'] = calibration_curve.get_calibrated_probs(df[cx_col_name].values)
    return df['calibr_pos'].sum() / len(df)

    