import pandas as pd


def get_data_types(data):
    all_columns = list(data.columns)
    date_preds = []
    categorical_preds = []
    numeric_preds = []

    for predictor in all_columns:
        if data[predictor].dtype == 'object':
            try:
                pd.to_datetime(data[predictor])
                date_preds.append(predictor)
            except Exception as e:
                categorical_preds.append(predictor)
                print('Ignore: ' + str(e))
        elif 'datetime' in str(data[predictor].dtype):
            date_preds.append(predictor)
        else:
            numeric_preds.append(predictor)

    print('=====================================')
    print('Dataset contains following data types')
    print('Date variables:', '[', len(date_preds), ']')
    print(date_preds)
    print('-------------------------------------')
    print('Categorical variables:', '[', len(categorical_preds), ']')
    print(categorical_preds)
    print('-------------------------------------')
    print('Numeric variables:', '[', len(numeric_preds), ']')
    print(numeric_preds)
    print('=====================================')

    return date_preds, categorical_preds, numeric_preds


def get_missing(data):
    missing_cnt = 0
    for col in data.columns:
        if data[col].isna().sum() == 0:
            pass
        else:
            missing_cnt += data[col].isna().sum()
    missing_pct = missing_cnt / (len(data.columns) * len(data))
    missing_out = data.isna().sum()

    print('=====================================')
    print('Total missing cells: [', missing_cnt, ']')
    print('Percentage of missing cells: [', missing_pct, ']')
    print('=====================================')
    print('Count of missing cells per column:')
    print(missing_out)
    print('=====================================')
    print('-------------------------------------')


def solve_missing(data, MISSING_ACTION):

    message_out = []
    all_columns = list(data.columns)
    date_preds = []
    categorical_preds = []
    numeric_preds = []

    for predictor in all_columns:
        if data[predictor].dtype == 'object':
            try:
                pd.to_datetime(data[predictor])
                date_preds.append(predictor)
            except Exception as e:
                categorical_preds.append(predictor)
                print('Ignore: ' + str(e))
        elif 'datetime' in str(data[predictor].dtype):
            date_preds.append(predictor)
        else:
            numeric_preds.append(predictor)

    # len_missing_action = len(MISSING_ACTION)

    if "replaceAll" == MISSING_ACTION:
        message_out.append('Replacing missing values in all columns:')
        for col in all_columns:
            if data[col].isna().sum() > 0:
                if col in numeric_preds:
                    data[col].fillna(data[col].mean(), inplace=True)
                else:
                    data[col].fillna('REPLACED-Undefined', inplace=True)
                message_out.append(col)

    if "replaceNumeric" == MISSING_ACTION:
        message_out.append('Replacing missing values in NUMERIC columns:')
        for col in numeric_preds:
            if data[col].isna().sum() > 0:
                data[col].fillna(data[col].mean(), inplace=True)
                message_out.append(col)

    if "replaceCategorical" == MISSING_ACTION:
        message_out.append('Replacing missing values in CATEGORICAL columns:')
        for col in categorical_preds:
            if data[col].isna().sum() > 0:
                data[col].fillna('REPLACED-Undefined', inplace=True)
                message_out.append(col)

    if "None" == MISSING_ACTION:
        pass

    if "drop" == MISSING_ACTION:
        message_out.append('Dropping all columns with any missing value.')
        data.dropna(inplace=True)

    print(message_out)
    return data
