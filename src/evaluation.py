import sys
import pandas as pd
import numpy as np
import datetime as dt
import math
import streamlit as st

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from grouping.woe_grouping import Grouping
from grouping.woe_grouping import gini_normalized
from data_exploration.kbc_data_exploration import solve_missing

from src.settings import snowpark_session
from snowflake.snowpark.functions import udf, col, lit, is_null, iff, initcap

@st.cache_data
def evaluate(schema_name, table_name, date_column, target_column, missing_action, skip_gradient_boosting):
    '''
    Main execution code
    '''
    table_name_full = '"' + schema_name + '"."' + table_name + '"'
    data = snowpark_session.table(table_name_full).to_pandas()
    data.drop('_timestamp', axis=1, inplace=True)

    # Extract basic descriptive statistics
    row_cnt = data.shape[0]
    col_cnt = data.shape[1]
    missing_cnt = 0
    for col in data.columns:
        if data[col].isna().sum() == 0:
            pass
        else:
            missing_cnt += data[col].isna().sum()
    missing_pct = missing_cnt / (len(data.columns) * len(data))
    duplicates_cnt = len(data) - len(data.drop_duplicates())

    # Extract counts of numerical, categorical and date variables in dataset
    numerical_cnt = 0
    categorical_cnt = 0
    date_cnt = 0
    date_preds = []
    numeric_preds = []
    categorical_preds = []
    all_columns = list(data.columns)

    if target_column != "target":
        data["target"] = data[target_column]
        data.drop(target_column, axis=1, inplace=True)
        all_columns.remove(target_column)

    if date_column not in ["", "Date_Start"]:
        data["Date_Start"] = data[date_column]
        data.drop(date_column, axis=1, inplace=True)
        all_columns.remove(date_column)

    if 'target' in all_columns:
        all_columns.remove('target')

    for predictor in all_columns:
        if data[predictor].dtype == 'object':
            try:
                pd.to_datetime(data[predictor])
                date_cnt += 1
                date_preds.append(predictor)
            except Exception as e:
                categorical_cnt += 1
                categorical_preds.append(predictor)
        elif 'datetime' in str(data[predictor].dtype):
            date_cnt += 1
            date_preds.append(predictor)
        else:
            numerical_cnt += 1
            numeric_preds.append(predictor)

    # ####### OUT ##########
    # general_out - table contains basic descriptive statistics about the dataset
    general_out = pd.DataFrame(
        {
            "DateGenerated": [dt.datetime.now()],
            "RowCnt": [row_cnt],
            "ColumnCnt": [col_cnt],
            "MissingCnt": [missing_cnt],
            "MissingPct": [missing_pct],
            "DuplicatesCnt": [duplicates_cnt],
            "NumericalVariableCnt": [numerical_cnt],
            "NumericalVariableList": [str(numeric_preds)],
            "CategoricalVariableCnt": [categorical_cnt],
            "CategoricalVariableList": [str(categorical_preds)],
            "DateVariableCnt": [date_cnt],
            "DateVariableList": [str(date_preds)],
        }
    )

    #general_out.to_csv(tables_out_path + '/general.csv', index=False)
    snowpark_session.create_dataframe(general_out).write.mode("overwrite").save_as_table("general")

    # Extract more detailed information about individual columns
    description_out = data.describe(include='all').transpose()
    description_out['MissingCnt'] = 0
    for col in data.columns:
        if data[col].isna().sum() == 0:
            pass
        else:
            description_out.loc[col, 'MissingCnt'] = data[col].isna().sum()
    description_out['ColumnName'] = description_out.index

    variable_code_mapping = pd.DataFrame({'VariableName': [], 'Code': []})
    j = 0
    for variable in description_out['ColumnName'].unique():
        code = str(j)
        variable_code_mapping = variable_code_mapping.append({"VariableName": variable + '_WOE', "Code": code},
                                                                ignore_index=True)
        j += 1
    snowpark_session.create_dataframe(variable_code_mapping).write.mode("overwrite").save_as_table("variable_code_mapping")#to_csv(tables_out_path + '/variableCodeMapping.csv', index=False)

    # ####### OUT ##########
    # description_out - table contains more detailed statistics of individual columns
    description_out['VariableName'] = description_out['ColumnName'] + '_WOE'
    description_out = pd.merge(description_out, variable_code_mapping, on='VariableName', how='left')
    snowpark_session.create_dataframe(description_out).write.mode("overwrite").save_as_table("description")#to_csv(tables_out_path + '/description.csv', index=False)

    # ######### PROCESS THE DATA AND EVALUATE PREDICTIVE VALUE ############
    # WOE TRANSFORMATION
    # IF target is present, features WOE will be calculated and model will be built
    if 'target' in data.columns:
        data = data.dropna(subset=['target'])
        data['target'] = data['target'].astype(float)

        # Solve missing values
        data = solve_missing(data, missing_action)

        predictor_list = numeric_preds + categorical_preds
        col_target = 'target'

        grouping = Grouping(columns=sorted(numeric_preds),
                            cat_columns=sorted(categorical_preds),
                            group_count=5,
                            min_samples=100,
                            min_samples_cat=100)

        grouping.fit(data[predictor_list],
                        data[col_target], category_limit=10000)

        data_woe = grouping.transform(data, transform_to='woe')
        for column in data_woe.columns:
            original_column_name = column.replace('_WOE', '')
            del data[original_column_name]

        data_woe_out = data.join(data_woe)
        data_woe_out['row_num'] = data_woe_out.index

        if 'Date_Start' in data_woe_out.columns:
            data_woe_out_melt = data_woe_out.melt(id_vars=["target", "Date_Start", "row_num"],
                                                    var_name="VariableName", value_name="Value")
            data_woe_out_melt_group = data_woe_out_melt.groupby(['VariableName', 'Date_Start', 'Value'])[
                'target'].agg(EventSum='sum', EventCount='count').reset_index()
        else:
            data_woe_out['Date_Start'] = '1900-01-01'
            data_woe_out_melt = data_woe_out.melt(id_vars=["target", "Date_Start", "row_num"],
                                                    var_name="VariableName", value_name="Value")
            data_woe_out_melt_group = data_woe_out_melt.groupby(['VariableName', 'Date_Start', 'Value'])[
                'target'].agg(EventSum='sum', EventCount='count').reset_index()

        data_woe_out_melt_group = pd.merge(data_woe_out_melt_group, variable_code_mapping, on='VariableName',
                                            how='left')

        data_woe_out_melt_group[
            'Order'] = np.nan  # will be filled by second transformation after evaluating predictors

        snowpark_session.create_dataframe(data_woe_out_melt).write.mode("overwrite").save_as_table("data_woe")#to_csv(tables_out_path + '/DataWOETransform.csv', index=False)
        snowpark_session.create_dataframe(data_woe_out_melt_group).write.mode("overwrite").save_as_table("data_woe_group")#.to_csv(tables_out_path + '/DataWOETransformGroup.csv', index=False)

    else:
        variable_code_mapping = pd.DataFrame({"VariableName": [], "Code": []})
        data_woe_out_melt = pd.DataFrame(
            {"target": [], "Date_Start": [], "row_num": [], "VariableName": [], "Value": []})
        data_woe_out_melt_group = pd.DataFrame(
            {"VariableName": [], "Date_Start": [], "Value": [], "EventSum": [], "EventCount": [], "Code": [],
                "Order": []})

        snowpark_session.create_dataframe(data_woe_out_melt).write.mode("overwrite").save_as_table("data_woe")#.to_csv(tables_out_path + '/DataWOETransform.csv', index=False)
        snowpark_session.create_dataframe(data_woe_out_melt_group).write.mode("overwrite").save_as_table("data_woe_group")#.to_csv(tables_out_path + '/DataWOETransformGroup.csv', index=False)
        snowpark_session.create_dataframe(variable_code_mapping).write.mode("overwrite").save_as_table("variable_code_mapping")#.to_csv(tables_out_path + '/variableCodeMapping.csv', index=False)


    # Load -in- dataset
    data = data_woe_out_melt
    data = data.pivot_table(index=["target", "Date_Start", 'row_num'], columns='VariableName',
                            values='Value').reset_index()
    data.drop('row_num', axis=1, inplace=True)
    data_woe = data_woe_out_melt_group

    if 'target' in data.columns:
        col_target = 'target'
        col_date = 'Date_Start'

        predictor_list = list(data.columns)
        predictor_list.remove(col_target)
        predictor_list.remove(col_date)

        corr = data[predictor_list].corr()
        corr['variable'] = corr.index

        for i, row in corr.iterrows():
            corr.rename(columns={row['variable']:
                                    variable_code_mapping[variable_code_mapping['VariableName'] == row['variable']]
                                    ['Code'].values[0]}, inplace=True)

            corr.at[i, 'variable'] = \
                variable_code_mapping[variable_code_mapping['VariableName'] == row['variable']]['Code'].values[0]

        corr = corr.melt(id_vars=["variable"], var_name="VariableName", value_name="Value")
        # ####### OUT ##########
        # Correlation matrix
        snowpark_session.create_dataframe(corr).write.mode("overwrite").save_as_table("correlation")#.to_csv(tables_out_path + '/correlation.csv', index=False)

        # Split dataset to train and test
        data['DAY'] = pd.to_numeric(pd.to_datetime(data['Date_Start'], format='%Y-%m-%d').dt.strftime('%Y%m%d'))
        data['MONTH'] = data['DAY'].apply(lambda x: math.trunc(x / 100))
        data['MONTH_START'] = pd.to_datetime(data['MONTH'].map(str) + '01')
        train, test = train_test_split(data, test_size=0.3)

        try:

            gr = data.groupby('MONTH_START', axis=0)
            res = gr.apply(lambda x: pd.Series(data=(len(x), 1. * len(x[(x['target'] == 1)]) / len(x)),
                                                index=['count', 'badRate']))
            res.index = res.index.astype(str)

            gr = train.groupby('MONTH_START', axis=0)
            restrain = gr.apply(lambda x: pd.Series(data=(len(x), 1. * len(x[(x['target'] == 1)]) / len(x)),
                                                    index=['trainCount', 'trainBadRate']))
            restrain.index = restrain.index.astype(str)

            gr = test.groupby('MONTH_START', axis=0)
            restest = gr.apply(lambda x: pd.Series(data=(len(x), 1. * len(x[(x['target'] == 1)]) / len(x)),
                                                    index=['testCount', 'testBadRate']))
            restest.index = restest.index.astype(str)

            res_restrain = pd.merge(res, restrain, left_index=True, right_index=True)
            res_restrain_restest = pd.merge(res_restrain, restest, left_index=True, right_index=True)

            # ####### OUT ##########
            # monthlyBadRate - table with share of BADS (target == 1) in dataset
            res_restrain_restest['month'] = res_restrain_restest.index
            snowpark_session.create_dataframe(res_restrain_restest).write.mode("overwrite").save_as_table("monthly_bad_rate")#.to_csv(tables_out_path + '/monthlyBadRate.csv', index=False)

        except Exception as e:
            empty = pd.DataFrame({'count': [], 'badRate': [], 'trainCount': [], 'trainBadRate': [], 'testCount': [],
                                    'testBadRate': [], 'month': []})
            snowpark_session.create_dataframe(empty).write.mode("overwrite").save_as_table("monthly_bad_rate")#.to_csv(tables_out_path + '/monthlyBadRate.csv', index=False)
            

        # Drop DATE columns from the dataset to run binary classification
        if 'Date_Start' in train.columns:
            train = train.drop('Date_Start', axis=1)
        if 'MONTH' in train.columns:
            train = train.drop('MONTH', axis=1)
        if 'DAY' in train.columns:
            train = train.drop('DAY', axis=1)
        if 'MONTH_START' in train.columns:
            train = train.drop('MONTH_START', axis=1)

        if 'Date_Start' in test.columns:
            test = test.drop('Date_Start', axis=1)
        if 'MONTH' in test.columns:
            test = test.drop('MONTH', axis=1)
        if 'DAY' in test.columns:
            test = test.drop('DAY', axis=1)
        if 'MONTH_START' in test.columns:
            test = test.drop('MONTH_START', axis=1)

        # seperate the independent and target variable on training data
        train_x = train.drop(columns=['target'], axis=1)
        train_y = train['target']

        # seperate the independent and target variable on testing data
        test_x = test.drop(columns=['target'], axis=1)
        test_y = test['target']

        # build model
        model = XGBClassifier(n_estimators=500, objective='binary:logistic', gamma=0.1, max_depth=3, eta=0.1)
        model.fit(train_x, train_y, eval_set=[(test_x, test_y)], eval_metric='auc', early_stopping_rounds=25)

        # predict the target on the train dataset
        predict_train = model.predict(train_x)

        # Accuracy Score on train dataset
        accuracy_train = accuracy_score(train_y, predict_train)

        # predict the target on the test dataset
        predict_test = model.predict(test_x)

        # Accuracy Score on test dataset
        accuracy_test = accuracy_score(test_y, predict_test)

        # xgboost_result - result of basic XGBoost run
        model_params = dict(model.get_params())
        xgboost_result = pd.DataFrame({"ModelName": ["XGBoost"],
                                        "BaseScore": model_params['base_score'],
                                        "Gamma": model_params['gamma'],
                                        "MaxDepth": model_params['max_depth'],
                                        "N_estimators": model_params['n_estimators'],
                                        "Eta": model_params['eta'],
                                        "Objective": model_params['objective'],
                                        "ROC_AUCTrain": [roc_auc_score(train_y, predict_train)],
                                        "ROC_AUCTest": [roc_auc_score(test_y, predict_test)],
                                        "TrainGini": gini_normalized(train_y, predict_train),
                                        "TestGini": gini_normalized(test_y, predict_test),
                                        "AccuracyTrain": [accuracy_train],
                                        "AccuracyTest": [accuracy_test]
                                        })
        xgboost_result.write.mode("overwrite").save_as_table("xgboost_result")#.to_csv(tables_out_path + '/xgboostResult.csv', index=False)

        # top25predictors
        top25predictors = pd.DataFrame(
            {'features': test_x.columns, 'importance': model.feature_importances_}).\
            sort_values('importance', ascending=False).head(25)

        snowpark_session.create_dataframe(top25predictors).write.mode("overwrite").save_as_table("top_predictors")#.to_csv(tables_out_path + '/top25predictors.csv', index=False)
        top25predictors['order'] = np.arange(len(top25predictors))

        data_woe = data_woe.merge(top25predictors, left_on='VariableName', right_on='features', how='left')
        data_woe.drop(['Order', 'features', 'importance'], axis=1, inplace=True)
        data_woe = data_woe.rename(columns={'order': 'Order'})
        snowpark_session.create_dataframe(data_woe).write.mode("overwrite").save_as_table("data_woe_group")#.to_csv(tables_out_path + '/DataWOETransformGroup.csv', index=False)
        other_models = pd.DataFrame(
            {"ModelName": [], "Accuracy": [], "GiniTest": [], "BestParams": [], "BestScore": []})

        # ### LOGISTIC REGRESSION
        logreg = LogisticRegression()
        logreg.fit(train_x, train_y)
        y_pred = logreg.predict(test_x)

        acc_logreg = round(accuracy_score(y_pred, test_y) * 100, 2)
        logreg_y_pred = logreg.predict(test_x)
        logreg_gini = gini_normalized(test_y, logreg_y_pred)

        other_models = other_models.append(
            {'ModelName': 'LogReg', 'Accuracy': acc_logreg, 'GiniTest': logreg_gini, 'BestParams': '',
                'BestScore': ''}, ignore_index=True)

        # ### DECISION TREE
        decisiontree = DecisionTreeClassifier()
        dep = np.arange(1, 10)
        param_grid = {'max_depth': dep}

        clf_cv = GridSearchCV(decisiontree, param_grid=param_grid, cv=5)

        clf_cv.fit(train_x, train_y)
        clf_cv_best_param = clf_cv.best_params_
        clf_cv_best_score = clf_cv.best_score_ * 100

        clf_cv_y_pred = clf_cv.predict(test_x)
        acc_clf_cv = round(accuracy_score(clf_cv_y_pred, test_y) * 100, 2)
        clf_cv_gini = gini_normalized(test_y, clf_cv_y_pred)

        other_models = other_models.append(
            {'ModelName': 'DecisionTree', 'Accuracy': acc_clf_cv, 'GiniTest': clf_cv_gini,
                'BestParams': clf_cv_best_param, 'BestScore': clf_cv_best_score}, ignore_index=True)

        # ### Random Forest CLassifier
        random_forest = RandomForestClassifier()
        ne = np.arange(1, 20)
        param_grid = {'n_estimators': ne}

        rf_cv = GridSearchCV(random_forest, param_grid=param_grid, cv=5)
        rf_cv.fit(train_x, train_y)

        rf_cv_best_param = rf_cv.best_params_
        rf_cv_best_score = rf_cv.best_score_ * 100

        rf_cv_y_pred = rf_cv.predict(test_x)
        acc_rf_cv = round(accuracy_score(rf_cv_y_pred, test_y) * 100, 2)
        rf_cv_gini = gini_normalized(test_y, rf_cv_y_pred)

        other_models = other_models.append(
            {'ModelName': 'RandomForest', 'Accuracy': acc_rf_cv, 'GiniTest': rf_cv_gini,
                'BestParams': rf_cv_best_param, 'BestScore': rf_cv_best_score}, ignore_index=True)

        # ### GRADIENT BOOSTING
        if skip_gradient_boosting == 0:
            gbk = GradientBoostingClassifier()
            dep = np.arange(1, 6)
            param_grid = {'n_estimators': [100, 250, 500, 750, 1000], 'max_depth': dep, 'learning_rate':
                            [0.25, 0.15, 0.1, 0.05, 0.01]}

            gbk_cv = GridSearchCV(gbk, param_grid=param_grid, scoring='roc_auc', cv=5)
            gbk_cv.fit(train_x, train_y)

            gbk_cv_best_param = gbk_cv.best_params_
            gbk_cv_best_score = gbk_cv.best_score_ * 100

            gbk_cv_y_pred = gbk_cv.predict(test_x)
            acc_gbk_cv = round(accuracy_score(gbk_cv_y_pred, test_y) * 100, 2)
            gbk_cv_gini = gini_normalized(test_y, gbk_cv_y_pred)

            other_models = other_models.append(
                {'ModelName': 'GradientBoosting', 'Accuracy': acc_gbk_cv, 'GiniTest': gbk_cv_gini,
                    'BestParams': gbk_cv_best_param, 'BestScore': gbk_cv_best_score}, ignore_index=True)

        snowpark_session.create_dataframe(other_models).write.mode("overwrite").save_as_table("other_models")#.to_csv(tables_out_path + '/otherModels.csv', index=False)
    else:
        corr = pd.DataFrame({"variable": [], "variableName": [], "value": []})
        snowpark_session.create_dataframe(corr).write.mode("overwrite").save_as_table("correlation")#.to_csv(tables_out_path + '/correlation.csv', index=False)
        monthly_br = pd.DataFrame(
            {"count": [], "badRate": [], "trainCount": [], "trainBadRate": [], "testCount": [], "testBadRate": [],
                "month": []})
        snowpark_session.create_dataframe(monthly_br).write.mode("overwrite").save_as_table("monthly_bad_rate")#.to_csv('out/tables/monthlyBadRate.csv', index=False)
        xgboost_result = pd.DataFrame(
            {"ModelName": ["XGBoost"], "BaseScore": [], "Gamma": [], "MaxDepth": [], "N_estimators": [], "Eta": [],
                "Objective": [], "ROC_AUCTrain": [], "ROC_AUCTest": [], "TrainGini": [], "TestGini": [],
                "AccuracyTrain": [], "AccuracyTest": []})
        snowpark_session.create_dataframe(xgboost_result).write.mode("overwrite").save_as_table("xgboost_result")#.to_csv('out/tables/xgboost_result.csv', index=False)
        top25predictors = pd.DataFrame({"features": [], "importance": []})
        snowpark_session.create_dataframe(top25predictors).write.mode("overwrite").save_as_table("top_predictors")#.to_csv('out/tables/top25predictors.csv', index=False)
        other_models = pd.DataFrame(
            {"ModelName": [], "Accuracy": [], "GiniTest": [], "BestParams": [], "BestScore": []})
        snowpark_session.create_dataframe(other_models).write.mode("overwrite").save_as_table("other_models")#.to_csv('out/tables/other_models.csv', index=False)
