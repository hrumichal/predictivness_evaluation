import copy
import json
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import assert_all_finite
from sklearn.utils.validation import check_consistent_length, column_or_1d
from sklearn.metrics import roc_auc_score
from tqdm import tqdm_notebook as tqdm


def gini_l(a, p):
    gini_out = 2 * roc_auc_score(a, p) - 1
    return gini_out


def gini_normalized(y_actual, y_pred):
    """Simple normalized Gini based on Scikit-Learn's roc_auc_score"""

    # If the predictions y_pred are binary class probabilities
    if y_pred.ndim == 2:
        if y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]

    return gini_l(y_actual, y_pred) / gini_l(y_actual, y_actual)


def gini_grp(woe, share, def_rate):
    df = pd.concat([woe, share, def_rate], axis=1)
    df.columns = ['woe', 'share', 'def_rate']
    df = df[df['share'] > 0].sort_values('woe')
    df['bad'] = df['share'] * df['def_rate']
    df['good'] = df['share'] - df['bad']
    df['bad_pct'] = df['bad'] / df['bad'].sum()
    df['good_pct'] = df['good'] / df['good'].sum()
    df['cum_bad_pct_0'] = df['bad_pct'].cumsum()
    df['cum_bad_pct'] = (df['cum_bad_pct_0'] + df['cum_bad_pct_0'].shift(1).fillna(0)) / 2
    df['auc'] = df['good_pct'] * df['cum_bad_pct']
    return (df['auc'].sum() - 0.5) * 2


def woe(y, y_full, smooth_coef, w=None, w_full=None):
    """Weight of evidence

    Parameters
    ----------
    y : np.array or pandas.Series
        target in current category, should contain just {0, 1}
    y_full : np.array or pandas.Series
        whole target
    smooth_coef : float
        coefficient to avoid divizion by zero or log of zero

    Returns
    -------
    woe : float
    """

    # TODO: think about default value of smooth_coef
    if smooth_coef < 0:
        raise ValueError('Smooth_coef should be non-negative')
    y = column_or_1d(y)
    y_full = column_or_1d(y_full)
    if y.size > y_full.size:
        raise ValueError('Length of y_full should be >= length of y')
    if len(set(y) - {0, 1}) > 0:
        raise ValueError('y should consist just of {0,1}')
    if not np.array_equal(np.unique(y_full), [0, 1]):
        raise ValueError('y_full should consist of {0,1}, noth should be presented')
    if w is not None and y.size != w.size:
        raise ValueError('Size of y and w must be the same')
    if w_full is not None and y_full.size != w_full.size:
        raise ValueError('Size of y_full and w_full must be the same')
    if w is None:
        w = np.ones(len(y))
    if w_full is None:
        w_full = np.ones(len(y_full))
    if y.size == 0:
        return 0.
    woe = np.log((sum((1 - y) * w) / sum(w) + smooth_coef) / (sum(y * w) / sum(w) + smooth_coef)) - np.log(
        (sum((1 - y_full) * w_full) / sum(w_full) + smooth_coef) / (sum(y_full * w_full) / sum(w_full) + smooth_coef))
    return woe


def nwoe(y, y_full, group, group_full, smooth_coef, w=None, w_full=None):
    """Net weight of evidence

    Parameters
    ----------
    y : np.array or pandas.Series
        target in current category, should contain just {0, 1}
    y_full : np.array or pandas.Series
        whole target
    group : np.array or pandas.Series
        binary uplift group in current category
    group_full : np.array or pandas.Series
        whole group
    smooth_coef : float
        coefficient to avoid divizion by zero or log of zero
    w : np.array or pandas.Series
        sample weights
    """

    if smooth_coef < 0:
        raise ValueError('Smooth_coef should be non-negative')
    y = column_or_1d(y)
    y_full = column_or_1d(y_full)
    if y.size > y_full.size:
        raise ValueError('Length of y_full should be >= length of y')
    if group.size > group_full.size:
        raise ValueError('Length of group_full should be >= length of group')
    if len(set(y) - {0, 1}) > 0:
        raise ValueError('y should consist just of {0,1}')
    if not np.array_equal(np.unique(y_full), [0, 1]):
        raise ValueError('y_full should consist of {0,1}, noth should be presented')
    if y.size != group.size:
        raise ValueError('Size of y and group must be the same')
    if w is not None and y.size != w.size:
        raise ValueError('Size of y and w must be the same')
    if w_full is not None and y_full.size != w_full.size:
        raise ValueError('Size of y_full and w_full must be the same')
    if w is None:
        w = np.ones(len(y))
    if w_full is None:
        w_full = np.ones(len(y_full))
    if y.size == 0:
        return 0.

    nwoe = np.log((sum((1 - y[group == 1]) * w[group == 1]) / (sum(w[group == 1]) + 1) + smooth_coef) /
                  (sum(y[group == 1] * w[group == 1]) / (sum(w[group == 1]) + 1) + smooth_coef)) - np.log(
        (sum((1 - y_full[group_full == 1]) * w_full[group_full == 1]) / sum(w_full[group_full == 1]) + smooth_coef) /
        (sum(y_full[group_full == 1] * w_full[group_full == 1]) / sum(w_full[group_full == 1]) + smooth_coef)) - \
        (np.log((sum((1 - y[group == 0]) * w[group == 0]) / (sum(w[group == 0]) + 1) + smooth_coef) /
                (sum(y[group == 0] * w[group == 0]) / (sum(w[group == 0]) + 1) + smooth_coef)) -
         np.log((sum((1 - y_full[group_full == 0]) * w_full[group_full == 0]) /
                 sum(w_full[group_full == 0]) + smooth_coef) /
                (sum(y_full[group_full == 0] * w_full[group_full == 0]) /
                 sum(w_full[group_full == 0]) + smooth_coef)))

    return nwoe


def tree_based_grouping(x, y, group_count, min_samples, w=None, group=None):
    """Grouping using decision trees

    Parameters
    ----------
    x : np.array or pandas.Series
        array of values, shape (number of observations,)
    y : np.array or pandas.Series
        binary target, shape (number of observations,)
    group_count : int
        maximum number of groups
    min_samples : int
        minimum number of objects in leaf (observations in each group)
    group : np.array or pandas.Series
        array with binary uplift group (treatment or control)
    w : np.array or pandas.Series
        sample weights

    Returns
    -------
    array of split points (including -+inf)

    Notes
    -------
    x should not include nans
    """
    # TODO: add documentation about split points selected by DecisionTreeClassifier (Pavel email)
    check_consistent_length(x, y)
    x = column_or_1d(x)
    assert_all_finite(x)
    y = column_or_1d(y)

    if len(set(y) - {0, 1}) > 0:
        raise ValueError('y should consist just of {0,1}')

    notnan_mask = ~np.isnan(x)

    if w is not None:
        check_consistent_length(y, w)
        w = column_or_1d(w)[notnan_mask]

    x = x.reshape(x.shape[0], -1)  # (n,) -> (n, 1)

    if group is None:
        clf = DecisionTreeClassifier(max_leaf_nodes=group_count, min_samples_leaf=min_samples)
        clf.fit(x[notnan_mask], y[notnan_mask], sample_weight=w)
    else:
        import uplift.tree
        clf = uplift.tree.DecisionTreeClassifier(criterion='uplift_gini', max_leaf_nodes=group_count,
                                                 min_samples_leaf=min_samples)
        clf.fit(x[notnan_mask], y[notnan_mask], group[notnan_mask], sample_weight=w)

    # TODO: check why == 0 is needed
    final_bins = np.concatenate([np.array([-np.inf]),
                                 np.sort(clf.tree_.threshold[clf.tree_.feature == 0]),
                                 np.array([np.inf])])
    return _convert_to_proper_bin_dtype(x.dtype, final_bins)


def auto_group_continuous(x, y, group_count, min_samples, woe_smooth_coef, bins=None, w=None, group=None):
    """Auto grouping continuous features

    Returns
    -------
    bins - list of intervals,
    woes - array of woes,
    nan_woe - array of woes
    """

    notnan_mask = x.notnull()
    if w is not None:
        w_nna = w[notnan_mask]
    else:
        w_nna = None

    if bins is None:
        bins = tree_based_grouping(x[notnan_mask], y[notnan_mask], group_count, min_samples, w=w_nna,
                                   group=None if group is None else group[notnan_mask])

    # temporary DataFrame since we need both x and y in grouping / aggregation
    if w is not None:
        df = pd.DataFrame({'x': x, 'y': y, 'w': w})
    else:
        w1 = np.ones(len(x))
        df = pd.DataFrame({'x': x, 'y': y, 'w': w1})

    if group is not None:
        df['group'] = group

    df.loc[pd.isnull(df['y']), 'w'] = np.nan
    bin_indices = pd.cut(df[notnan_mask]['x'], bins=bins, right=False, labels=False)
    # sg Some values can be missing in new data
    woes = np.zeros(bins.shape[0] - 1)
    if group is None:
        new_woes = df.groupby(bin_indices).apply(
            lambda rows: woe(rows['y'], df['y'], woe_smooth_coef, w=rows['w'], w_full=df['w'])).to_dict()
        np.put(woes, list(new_woes.keys()), list(new_woes.values()))
        nan_woe = woe(df[~notnan_mask]['y'], df['y'], woe_smooth_coef, w=df[~notnan_mask]['w'], w_full=df['w'])
    else:
        new_woes = df.groupby(bin_indices).apply(
            lambda rows: nwoe(rows['y'], df['y'], rows['group'], df['group'], woe_smooth_coef, w=rows['w'],
                              w_full=df['w'])).to_dict()
        np.put(woes, list(new_woes.keys()), list(new_woes.values()))
        nan_woe = nwoe(df[~notnan_mask]['y'], df['y'], df[~notnan_mask]['group'], df['group'], woe_smooth_coef,
                       w=df[~notnan_mask]['w'], w_full=df['w'])

    return bins, woes, nan_woe


def auto_group_categorical(x, y, group_count, min_samples, min_samples_cat, woe_smooth_coef, bins=None, w=None,
                           group=None):
    """Auto grouping categorical features

    Returns
    -------
    bins - dict(value->group number),
    woes - array of woes,
    unknown_woe = 0
    """
    # temporary DataFrame since we need both x and y in grouping / aggregation
    # print('auto_group_categorical')
    if x.dtype.name == 'category':
        x = x.astype(x.cat.categories.dtype.name)

    if w is not None:
        df = pd.DataFrame({'x': x, 'y': y, 'w': w})
    else:
        w1 = np.ones(len(x))
        df = pd.DataFrame({'x': x, 'y': y, 'w': w1})

    if group is not None:
        df['group'] = group

    df.loc[pd.isnull(df['y']), 'w'] = np.nan
    df['wy'] = df['w'] * df['y']

    if bins is None:
        if group is None:
            stats = df.groupby('x').apply(
                lambda rows: pd.Series(index=['cnt', 'cnt_bads'], data=[rows['w'].sum(), rows['wy'].sum()]))
            stats['event_rate'] = np.nan
            stats.loc[stats['cnt'] > 0, 'event_rate'] = stats['cnt_bads'] / stats['cnt']
            nan_stat = pd.Series(index=['cnt', 'cnt_bads'],
                                 data=[df[df.x.isnull()]['w'].sum(), df[df.x.isnull()]['wy'].sum()])
            if nan_stat['cnt'] > 0:
                nan_stat['event_rate'] = nan_stat['cnt_bads'] / nan_stat['cnt']
            else:
                nan_stat['event_rate'] = np.nan
        else:
            stats = df.groupby('x').apply(
                lambda rows: pd.Series(index=['cnt', 'cnt_0', 'cnt_1', 'cnt_bads_0', 'cnt_bads_1'],
                                       data=[rows['w'].sum(),
                                             rows[rows['group'] == 0]['w'].sum(),
                                             rows[rows['group'] == 1]['w'].sum(),
                                             rows[rows['group'] == 0]['wy'].sum(),
                                             rows[rows['group'] == 1]['wy'].sum(),
                                             ]))
            stats['event_rate'] = np.nan
            stats.loc[(stats['cnt_0'] > 0) & (stats['cnt_1'] > 0), 'event_rate'] = \
                stats['cnt_bads_1'] / stats['cnt_1'] - stats['cnt_bads_0'] / stats['cnt_0']

            nan_stat = pd.Series(index=['cnt', 'cnt_0', 'cnt_1', 'cnt_bads_0', 'cnt_bads_1'],
                                 data=[df[df.x.isnull()]['w'].sum(),
                                       df[df.x.isnull() & (group == 0)]['w'].sum(),
                                       df[df.x.isnull() & (group == 1)]['w'].sum(),
                                       df[df.x.isnull() & (group == 0)]['wy'].sum(),
                                       df[df.x.isnull() & (group == 1)]['wy'].sum(),
                                       ])
            if nan_stat['cnt_0'] > 0 and nan_stat['cnt_1'] > 0:
                nan_stat['event_rate'] = nan_stat['cnt_bads_1'] / \
                                         nan_stat['cnt_1'] - nan_stat['cnt_bads_0'] / nan_stat['cnt_0']
            else:
                nan_stat['event_rate'] = np.nan

        # (DG) rare mask doesn't consider treatment and control counts separetly (make review leater)
        rare_mask = stats['cnt'] < min_samples_cat
        rare_values = stats[rare_mask].index.values
        rare_df = df.join(pd.DataFrame(index=rare_values), on='x', how='inner')
        rare_w = rare_df['w'].values
        rare_wy = rare_df['wy'].values
        # sg!!!
        # cat -> statistically significant event-rate
        mapping = stats[~rare_mask]['event_rate'].to_dict()

        if group is None:
            if nan_stat['cnt'] >= min_samples_cat:
                mapping[np.nan] = nan_stat['event_rate']
            elif nan_stat['cnt'] > 0:
                rare_values = np.append(rare_values, np.nan)
                rare_w = np.append(rare_w, df[df.x.isnull()].w.values)
                rare_wy = np.append(rare_wy, df[df.x.isnull()].wy.values)

            mapping.update({v: rare_wy.sum() / rare_w.sum() for v in rare_values})
        else:
            rare_group = rare_df['group'].values

            if nan_stat['cnt'] >= min_samples_cat and nan_stat['cnt_0'] > 0 and nan_stat['cnt_1'] > 0:
                mapping[np.nan] = nan_stat['event_rate']
            elif nan_stat['cnt'] > 0:  # nan_stat['cnt'] > 0:
                rare_values = np.append(rare_values, np.nan)
                rare_w = np.append(rare_w, df[df.x.isnull()].w.values)
                rare_wy = np.append(rare_wy, df[df.x.isnull()].wy.values)
                rare_group = np.append(rare_group, df[df.x.isnull()].group.values)

            if rare_w[rare_group == 1].sum() > 0 and rare_w[rare_group == 0].sum() > 0 and \
                    rare_w.sum() >= min_samples_cat:
                mapping.update({v: rare_wy[rare_group == 1].sum() / rare_w[rare_group == 1].
                               sum() - rare_wy[rare_group == 0].sum() / rare_w[rare_group == 0].sum()
                                for v in rare_values})
            else:
                mapping.update({v: 0. for v in rare_values})

        # new continuous column
        x2 = df.x.replace(mapping)

        bins = tree_based_grouping(x2, y, group_count, min_samples, w=w, group=group)

        # mapping: cat -> ER
        # bins: ER [-inf, 0.1, 0.34,+inf]]

        # sg - rewrite - now duplicated functionality with "else" below
        bin_indices = pd.cut(x2, bins=bins, right=False, labels=False)

        if group is None:
            woes = df.groupby(bin_indices).apply(lambda rows: woe(
                rows['y'], df['y'], woe_smooth_coef, w=rows['w'], w_full=df['w'])).values
        else:
            woes = df.groupby(bin_indices).apply(lambda rows: nwoe(
                rows['y'], df['y'], rows['group'], df['group'], woe_smooth_coef, w=rows['w'], w_full=df['w'])).values

        # cat -> group number Series
        groups = pd.cut(pd.Series(mapping), bins=bins, right=False, labels=False)
        bins = groups.to_dict()
        if nan_stat['cnt'] == 0:
            # nan_group = pd.cut([m[np.nan]], bins=bins, right=False, labels=False)[0]
            # new group for nan with WOE=0.
            nan_group = groups.max() + 1
            woes = np.append(woes, [0.])
            bins[np.nan] = nan_group

        # WOE for values that are not present in the training set
        unknown_woe = 0

    else:
        # sg duplication here!
        # this branch was added to support "recalc WOEs on fixed bins (splits)" mode
        # {1: 2, 3: 2, 4: 0}
        df['tmp'] = np.nan
        for cat, g in bins.items():
            if type(cat) == float and np.isnan(cat):
                df.loc[df.x.isnull(), 'tmp'] = g
            else:
                df.loc[df.x == cat, 'tmp'] = g
        # sg Some values can be missing in new data
        woes = np.zeros(len(bins))

        if group is None:
            new_woes = df.groupby(df['tmp']).apply(lambda rows: woe(
                rows['y'], df['y'], woe_smooth_coef, w=rows['w'], w_full=df['w'])).to_dict()
        else:
            new_woes = df.groupby(df['tmp']).apply(lambda rows: nwoe(
                rows['y'], df['y'], rows['group'], df['group'], woe_smooth_coef, w=rows['w'], w_full=df['w'])).to_dict()

        np.put(woes, list(new_woes.keys()), list(new_woes.values()))
        unknown_woe = 0

    return bins, woes, unknown_woe


def _event_rates(x, y, bins, w=None):
    """Event rates

    Returns
    -------
    event rates for the given bins
    """
    values = pd.Series([np.nan] * (len(bins) - 1), index=range(1, len(bins)))
    if w is None:
        values.update(y.groupby(np.digitize(x, bins)).mean())
        result = values.values
    else:
        wy = w * y
        df = pd.DataFrame({'x': x, 'y': y, 'w': w, 'wy': wy})
        df = df.groupby(np.digitize(df['x'], bins)).sum()
        values.update(df['wy'] / df['w'])
        result = values.values
    return result


def _convert_to_proper_bin_dtype(data_type, target):
    '''
    Converts `target` to proper dtype based on `data_type`
    float16/32 -> float16/32
    others -> float64
    '''
    if data_type is np.dtype(np.float16):
        return np.float16(target)
    elif data_type is np.dtype(np.float32):
        return np.float32(target)
    else:
        return target


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class NumpyJSONDecoderOld(json.JSONDecoder):
    """
    For compatibility
    """

    def __init__(self):
        super(NumpyJSONDecoderOld, self).__init__(object_hook=self.hook)

    def hook(self, d):
        if 'bins' not in d:  # we need to apply changes only to second level of JSON "objects"
            return d
        if isinstance(d['bins'], list):
            d['bins'] = np.array(d['bins'])
        d['woes'] = np.array(d['woes'])
        return d


class NumpyJSONDecoder(json.JSONDecoder):
    """
    TODO: rename, it's not universal numpy decoder
    """

    def __init__(self):
        super(NumpyJSONDecoder, self).__init__(object_hook=self.hook)

    def hook(self, d):
        if 'bins' not in d and 'cat_bins' not in d:  # we need to apply changes only to second level of JSON "objects"
            return d
        if 'bins' in d:
            d['bins'] = np.array(d['bins'])
        d['woes'] = np.array(d['woes'])
        return d


class Grouping(BaseEstimator, TransformerMixin):
    """Grouping

    Attributes:
    bins_data: list of dict with keys:
               bins
               woes
               nan_woe
    """

    def __init__(self, columns, cat_columns, group_count=3, min_samples=1, min_samples_cat=1, woe_smooth_coef=0.001,
                 filename=None):
        """
        Parameters
        ----------
        columns : list
            continuous columns
        cat_columns : list
            categorical columns
        group_count : int
            maximum number of groups
        min_samples : int
            minimum number of objects in leaf (observations in each group)
        min_samples_cat : int
            minimal number of samples in category to trust event rate
        woe_smooth_coef : float
            smooth coefficient
        filename
        """

        self.columns = columns
        self.cat_columns = cat_columns
        self.group_count = group_count
        self.min_samples = min_samples
        self.min_samples_cat = min_samples_cat  # sg
        self.woe_smooth_coef = woe_smooth_coef
        self.filename = filename  # not needed
        if filename is not None:
            self.load(filename)

    def transform_old(self, X):
        """
        Applies grouping

        Parameters
        ----------
        X : pandas.DataFrame
        """
        # TODO: now disabled since used in InteractiveGrouping
        # check_is_fitted(self, 'bins_data_')
        X_woe = pd.DataFrame(columns=self.columns)

        for column in self.columns:
            bin_data = self.bins_data_[column]
            x_woe = pd.cut(X[column], bin_data['bins'],
                           right=False, labels=False)
            x_woe[x_woe.notnull()] = bin_data['woes'][x_woe[x_woe.notnull()].astype(int)]
            # if bin_data['has_nan']:
            x_woe[x_woe.isnull()] = bin_data['nan_woe']
            X_woe[column] = x_woe

        for column in self.cat_columns:
            bin_data = self.bins_data_[column]
            # replacement for categorical values
            x_woe = X[column].copy()
            x_woe[~pd.isnull(X[column])] = X.loc[~pd.isnull(X[column]), column].replace(
                {value: bin_data['woes'][group] for value, group in bin_data['bins'].items()})
            known = ~X[column].isnull() & X[column].isin(bin_data['bins'])

            # replacement for NaN values if there were part of the binning.
            for k in bin_data['bins'].keys():
                # if (pd.isnull(k)) or (k == 'NaN') or (k == 'nan'):
                if (pd.isnull(k)):
                    known = known | x_woe.isnull()
                    x_woe.fillna(bin_data['woes'][bin_data['bins'][k]], inplace=True)
            # replacement for unknown values
            x_woe[~known] = bin_data['unknown_woe']
            X_woe[column] = pd.to_numeric(x_woe)
        return X_woe

    def get_dummies(self, columns_to_transform=None):

        if columns_to_transform is not None:
            for column in columns_to_transform:
                if column not in self.columns + self.cat_columns:
                    raise ValueError(f'Column {column} not in grouping.')
            cols_num = [col for col in columns_to_transform if col in self.columns]
            cols_cat = [col for col in columns_to_transform if col in self.cat_columns]
        else:
            cols_num = self.columns
            cols_cat = self.cat_columns
            columns_to_transform = cols_num + cols_cat

        dummies = {}
        suffix = '_DMY'

        for name in columns_to_transform:
            if name in cols_num:
                bin_data = self.bins_data_[name]
                dummy_vars = []
                for i in range(len(bin_data['woes'])):
                    dummy_name = f'{name}{suffix}_{i}'
                    dummy_vars.append(dummy_name)
                dummy_name = f'{name}{suffix}_NaN'
                dummy_vars.append(dummy_name)
                dummies[name] = dummy_vars

            if name in cols_cat:
                bin_data = self.bins_data_[name]
                dummy_vars = []
                for i in range(len(bin_data['woes'])):
                    dummy_name = f'{name}{suffix}_{i}'
                    dummy_vars.append(dummy_name)
                dummy_name = f'{name}{suffix}_Unknown'
                dummy_vars.append(dummy_name)
                dummies[name] = dummy_vars

        return dummies

    def transform(self, data, transform_to='woe', columns_to_transform=None, progress_bar=False):
        """
        Performs transformation of `data` based on `transform_to` parameter and adds `suffix` to column names.

        Parameters
        ----------
        transform_to : possible values `woe`,`shortnames`,`group_number`,`dummy`
        """

        if columns_to_transform is not None:
            for column in columns_to_transform:
                if column not in self.columns + self.cat_columns:
                    raise ValueError('Column {} not in grouping.'.format(column))
            cols_num = [col for col in columns_to_transform if col in self.columns]
            cols_cat = [col for col in columns_to_transform if col in self.cat_columns]
        else:
            cols_num = self.columns
            cols_cat = self.cat_columns
            columns_to_transform = cols_num + cols_cat

        if transform_to not in {'woe', 'shortnames', 'group_number', 'dummy'}:
            raise ValueError("'{0}' is not a valid transform_to value "
                             "('woe', 'shortnames', 'group_number', 'dummy').".format(transform_to))
        else:
            suffix_dict = {'woe': '_WOE', 'shortnames': '_RNG', 'group_number': '_GRP', 'dummy': '_DMY'}
            suffix = suffix_dict[transform_to]

        if progress_bar:
            iterator = tqdm(data[columns_to_transform].iteritems(), total=len(columns_to_transform), leave=True,
                            unit='cols')
        else:
            iterator = data[columns_to_transform].iteritems()

        if transform_to != 'dummy':
            data_woe = pd.DataFrame(columns=cols_num + cols_cat)
        else:
            data_woe = pd.DataFrame(index=data.index)

        for name, column in iterator:
            # print(name)
            if progress_bar:
                iterator.set_description(desc=name, refresh=True)
            if name in cols_num:
                bin_data = self.bins_data_[name]

                if transform_to == 'woe':
                    # use standard woe values
                    target_values = bin_data['woes'].astype(np.float32)
                    target_nan = bin_data['nan_woe']

                elif transform_to == 'shortnames':
                    # use internaval as shortnames eg. (-inf,1.35]
                    target_values = ['[{:.3f}, {:.3f})'.format(bin_data['bins'][i], bin_data['bins'][i + 1]) for i in
                                     range(len(bin_data['bins']) - 1)]
                    target_nan = 'NaN'

                elif transform_to == 'group_number':
                    # use group number
                    target_values = [i for i in range(len(bin_data['bins']) - 1)]
                    target_nan = len(bin_data['bins']) - 1

                if transform_to != 'dummy':
                    tmp = pd.cut(column, bin_data['bins'], right=False, labels=False)
                    map_dict = {np.nan: target_nan, **{i: target_values[i] for i in range(len(target_values))}}
                    data_woe[name] = tmp.map(map_dict)

                else:  # create dummy variables
                    tmp = pd.cut(column, bin_data['bins'], right=False, labels=False)
                    for i in range(len(bin_data['woes'])):
                        dummy_name = self.get_dummies(columns_to_transform=[name])[name][i]
                        data_woe[dummy_name] = 0
                        data_woe.loc[tmp == i, dummy_name] = 1
                    dummy_name = self.get_dummies(columns_to_transform=[name])[name][-1]
                    data_woe[dummy_name] = 0
                    data_woe.loc[pd.isnull(tmp), dummy_name] = 1

            if name in cols_cat:
                bin_data = self.bins_data_[name]

                if transform_to == 'woe':
                    # use standard woe values
                    target_values = bin_data['woes'].astype(np.float32)
                    target_nan = bin_data['unknown_woe']

                    map_dict = {k: target_values[v] for k, v in bin_data['bins'].items()}
                    data_woe[name] = column.map(map_dict).astype(np.float32).fillna(target_nan)

                elif transform_to == 'shortnames':
                    # use concatenated values as group names
                    groups = [[] for i in range(len(bin_data['woes']))]
                    for value, group in bin_data['bins'].items():
                        groups[group].append(str(value))
                    # cut these names to 40chars
                    target_values = [','.join(s)[:40] for s in groups]
                    # add numbering to potential duplicate group names
                    for target_name, count in Counter(target_values).items():
                        if count > 1:
                            for suf in [' '] + list(range(1, count)):
                                target_values[target_values.index(target_name)] = target_name + str(suf)
                    target_nan = 'Unknown'

                    map_dict = {k: target_values[v] for k, v in bin_data['bins'].items()}
                    data_woe[name] = column.astype(str).map(map_dict).fillna(target_nan)

                elif transform_to == 'group_number':
                    # use group number
                    s = set(bin_data['bins'].values())
                    target_values = [i for i in range(len(s))]
                    target_nan = len(s)

                    map_dict = {k: target_values[v] for k, v in bin_data['bins'].items()}
                    data_woe[name] = column.map(map_dict).astype(np.float32).fillna(target_nan)

                elif transform_to == 'dummy':  # create dummy variables
                    s = set(bin_data['bins'].values())
                    target_values = [i for i in range(len(s))]
                    target_nan = len(s)
                    map_dict = {k: target_values[v] for k, v in bin_data['bins'].items()}
                    tmp = column.map(map_dict).astype(np.float32).fillna(target_nan)
                    for i in range(len(bin_data['woes'])):
                        dummy_name = self.get_dummies(columns_to_transform=[name])[name][i]
                        data_woe[dummy_name] = 0
                        data_woe.loc[tmp == target_values[i], dummy_name] = 1
                    dummy_name = self.get_dummies(columns_to_transform=[name])[name][-1]
                    data_woe[dummy_name] = 0
                    data_woe.loc[tmp == target_nan, dummy_name] = 1

        if transform_to == 'group_number':
            data_woe = data_woe.astype(np.int32)
        elif transform_to == 'dummy':
            data_woe = data_woe.astype(np.int16)
        elif transform_to == 'woe':
            data_woe = data_woe.astype(np.float32)

        if transform_to != 'dummy':
            renaming = {col: col + suffix for col in data_woe.columns}
        else:
            renaming = {col: col for col in data_woe.columns}

        return data_woe.rename(renaming, axis='columns')

    def saveOld(self, filename):
        # check_is_fitted(self, 'bins_data_')
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.bins_data_, f, ensure_ascii=False,
                      cls=NumpyJSONEncoder)

    def loadOld(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            self.bins_data_ = json.load(f, cls=NumpyJSONDecoderOld)
        # ugly workaround to have nan properly assigned
        for v, g in self.bins_data_.items():
            changenan = False
            nanval = 0
            if isinstance(g['bins'], dict):
                for c, b in g['bins'].items():
                    if c == 'NaN':
                        changenan = True
                        nanval = b
                if changenan:
                    g['bins'][np.nan] = nanval
                    del g['bins']['NaN']

    # sg
    def save(self, filename):
        # check_is_fitted(self, 'bins_data_')
        tmp = copy.deepcopy(self.bins_data_)
        for k, v in tmp.items():
            if isinstance(v['bins'], dict):
                v['cat_bins'] = list(v['bins'].items())
                del v['bins']

        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(tmp,
                      file,
                      ensure_ascii=False,
                      cls=NumpyJSONEncoder,
                      indent=2)

    def load(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            self.bins_data_ = json.load(f, cls=NumpyJSONDecoder)
            # display(self.bins_data_)
            for k, v in self.bins_data_.items():
                if 'cat_bins' in v:
                    v['bins'] = dict(v['cat_bins'])
                    del v['cat_bins']
                    # NaNs workaround
                    if "NaN" in v['bins']:
                        v[np.nan] = v['NaN']
                        del v['NaN']

                if 'dtype' in v.keys():
                    if v['dtype'] == 'float16':
                        v['bins'] = v['bins'].astype(np.float16)
                    elif v['dtype'] == 'float32':
                        v['bins'] = v['bins'].astype(np.float32)

    def _auto_grouping(self, x, y, w=None, group=None):
        # print(x.name)

        if not ((type(x) is pd.Series) or (type(x) is np.ndarray and x.ndim == 1)):
            raise ValueError("x should be a pandas.Series or 1d numpy.array")

        if not ((type(y) is pd.Series) or (type(y) is np.ndarray and x.ndim == 1)):
            raise ValueError("y should be a pandas.Series")

        if x.name in self.columns:
            bins, woes, nan_woe = auto_group_continuous(
                x, y, self.group_count, self.min_samples, self.woe_smooth_coef,
                bins=self.bins_data_[x.name]['bins'] if x.name in self.bins_data_ else None, w=w, group=group)
        else:
            bins, woes, unknown_woe = auto_group_categorical(
                x, y, self.group_count, self.min_samples, self.min_samples_cat, self.woe_smooth_coef,
                bins=self.bins_data_[x.name]['bins'] if x.name in self.bins_data_ else None, w=w, group=group)

        bin_data = {'bins': bins, 'woes': woes, }

        if x.name in self.columns:
            bin_data['nan_woe'] = nan_woe
        else:
            bin_data['unknown_woe'] = unknown_woe

        bin_data['dtype'] = x.dtype.name
        self.bins_data_[x.name] = bin_data

    def fit(self, X, y, w=None, progress_bar=False, category_limit=100):
        """
        Makes automatic grouping

        Parameters
        ----------
        X : pandas.DataFrame

        y : pandas.Series or np.array

        w : np.array or pandas.Series
            sample weights
        progress_bar : bool
            progress bar
        category_limit : int

        """

        if type(X) != pd.DataFrame:
            raise ValueError('X should be DataFrame')
        check_consistent_length(X, y)
        y = column_or_1d(y)

        if np.any(X.columns.duplicated()):
            duplicities = [col_name for col_name, duplicated in zip(X.columns, X.columns.duplicated()) if duplicated]
            raise ValueError(
                f"Columns {list(dict.fromkeys(duplicities))} are duplicated in your Dataset.")

        for name, column in X[self.columns].iteritems():
            if np.any(np.isinf(column.values)):
                raise ValueError(f'Column {name} containes non-finite values.')

        if w is not None:
            check_consistent_length(w, y)
            w = column_or_1d(w).astype('float64')
            w[pd.isnull(y)] = np.nan

        for name, column in X[self.cat_columns].iteritems():
            if column.nunique() > category_limit:
                raise ValueError('Column {0} has more than {1} unique values. '
                                 'Large number of unique values might cause memory issues. '
                                 'This limit can be set with parameter `category_limit.'.format(name, category_limit))

        if not hasattr(self, 'bins_data_'):
            self.bins_data_ = {}
        # So we will keep already trained bins if they exists (loaded from file or fitted before)

        if progress_bar:
            iterator = tqdm(self.columns + self.cat_columns, leave=True, unit='cols')
        else:
            iterator = self.columns + self.cat_columns

        for column in iterator:
            if progress_bar:
                iterator.set_description(desc=column, refresh=True)
            self._auto_grouping(X[column], y, w)
        return self  # sg

    def fit_uplift(self, X, y, group, w=None):
        """Makes automatic grouping

        Parameters
        ----------
        X : pandas.DataFrame
            dataframe with features
        y : pandas.Series
            Series with target
        group : pandas.Series
            Series with uplift group (treatment or control)
        w : np.array or pandas.Series
            sample weights
        """

        if type(X) != pd.DataFrame:
            raise ValueError('X should be DataFrame')
        check_consistent_length(X, y)
        y = column_or_1d(y)

        if w is not None:
            check_consistent_length(w, y)
            w = column_or_1d(w).astype('float64')
            w[pd.isnull(y)] = np.nan

        if not hasattr(self, 'bins_data_'):
            self.bins_data_ = {}

        iterator = self.columns + self.cat_columns

        for column in iterator:
            self._auto_grouping(X[column], y, w, group)

        return self

    def export_dictionary(self, suffix="_WOE", interval_edge_rounding=3, woe_rounding=5):
        '''
        Returns a dictionary with (woe:bin/values) pairs for fitted predictors.
        Numerical predictors are in this format:
            round(woe): "[x, y)"
        Categorical predictors are in this format:
            round(woe): ["AA","BB","CC","Unknown"]

        Args:
        interval_edge_rounding (int, optional):
        woe_rounding (int, optional):


        Example:
        {'Numerical_1_WOE': {-0.77344: '[-inf, 0.093)',
                         -0.29478: '[0.093, 0.248)',
                          0.16783: '[0.248, 0.604)',
                          0.86906: '[0.604, 0.709)',
                          1.84117: '[0.709, inf)',
                              0.0: 'NaN'},
        'Categorical_1_WOE': { 0.34995: 'EEE, FFF, GGG, III',
                           0.03374: 'HHH',
                          -0.16117: 'CCC, DDD',
                          -0.70459: 'BBB',
                          -0.95404: 'AAA',
                               0.0: 'nan, Unknown'}}


        '''

        woe_dictionary = dict()
        # iterate over numerical predictors
        for col in self.columns:

            # skip vars that are not fitted
            if col not in self.bins_data_.keys():
                continue

            grouping_data = self.bins_data_[col]
            woe_dictionary[col + suffix] = {}

            # go over other intervals
            bins = grouping_data['bins']
            woes = grouping_data['woes']
            intervals = list(zip(bins, bins[1:]))
            for woe, (lower, upper) in zip(woes, intervals):
                woe_dictionary[col + suffix][round(np.float32(woe),
                                                   woe_rounding)] = \
                    f"[{lower:.{interval_edge_rounding}f}, {upper:.{interval_edge_rounding}f})"

            # NaN WOE handled separately
            nan_woe = round(np.float32(grouping_data['nan_woe']), woe_rounding)
            if nan_woe in woe_dictionary[col + suffix].keys():
                woe_dictionary[col + suffix][nan_woe] += ' NaN'
            else:
                woe_dictionary[col + suffix][nan_woe] = 'NaN'

        # interate over categorical predictors
        for col in self.cat_columns:

            # skip vars that are not fitted
            if col not in self.bins_data_.keys():
                continue

            grouping_data = self.bins_data_[col]
            woe_dictionary[col + suffix] = {}
            bins = grouping_data['bins']
            woes = grouping_data['woes']

            groups = [list() for _ in woes]
            for value, bin_ in bins.items():
                groups[bin_].append(str(value))

            for woe, group in zip(woes, groups):
                woe_dictionary[col + suffix][round(np.float32(woe), woe_rounding)] = group

            unknown_woe = round(np.float32(grouping_data['unknown_woe']), woe_rounding)
            if unknown_woe in woe_dictionary[col + suffix].keys():
                woe_dictionary[col + suffix][unknown_woe].append("Unknown")
            else:
                woe_dictionary[col + suffix][unknown_woe] = ["Unknown"]

        return woe_dictionary


class Wrapper(object):
    """Very simple wrapper to make binding between elems and textboxes easier"""

    def __init__(self, val):
        self.val = val


class Context(object):
    def __init__(self, column, grouping):
        self.column = column  # column
        self.grouping = grouping

    @property
    def x(self):
        return self.grouping.train_t[self.column]

    @property
    def y(self):
        return self.grouping.train_t[self.grouping.target_column]

    @property
    def weight(self):
        if self.grouping.w_column is not None:
            w = self.grouping.train_t[self.grouping.w_column].copy()
            w[pd.isnull(self.grouping.train_t[self.grouping.target_column])] = np.nan
            return w
        else:
            w = pd.Series(data=np.ones(len(self.grouping.train_t[self.column])),
                          index=self.grouping.train_t[self.column].index)
            w[pd.isnull(self.grouping.train_t[self.grouping.target_column])] = np.nan
            return w

    @property
    def has_nan(self):
        return self.x.isnull().any()

    def update(self, tab_change=False):
        # print('update', tab_change)
        if tab_change:
            self.grouping.fig.clear()
            self._create_plots()

        valid = self.validate()

        self.grouping.validate_all()

        if valid:
            self._update_data()
        self._update_form(valid)
        self.grouping.fig.canvas.draw()
