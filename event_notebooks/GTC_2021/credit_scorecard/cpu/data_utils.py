from woesc_utils import *


def preprocess_vehicle_data(dataset, id_vars, targ_var):
    ## Sort data by unique client identifier
    dataset.sort_values(id_vars, inplace=True)
    dataset.reset_index(drop=True, inplace=True)

    ## Make the target variable the second column in the dataframe
    targets = dataset.pop(targ_var)
    dataset.insert(1, targ_var, targets)

    ## Replace periods in variable names with underscores 
    new_cols = [sub.replace('.', '_') for sub in dataset.columns] 
    dataset.rename( columns=dict(zip(dataset.columns, new_cols)), inplace=True)

    ## Specify variables that should be treated as categorical and convert them to character strings (non-numeric)
    cat_vars = [ 'branch_id', 'supplier_id', 'manufacturer_id', 'Current_pincode_ID', 'State_ID', 'Employee_code_ID'
                , 'Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag', 'Passport_flag']
    dataset[cat_vars] = dataset[cat_vars].fillna('')
    dataset[cat_vars] = dataset[cat_vars].applymap(str)

    ## Strategically add some missing data 
    ## Note: There is no bureau data for more than half of the records
    no_bureau = (dataset.PERFORM_CNS_SCORE_DESCRIPTION == 'No Bureau History Available')
    dataset.loc[no_bureau, 'PERFORM_CNS_SCORE_DESCRIPTION'] = ''
    bureau_vars = [ 'PERFORM_CNS_SCORE', 'PRI_NO_OF_ACCTS', 'PRI_ACTIVE_ACCTS', 'PRI_OVERDUE_ACCTS'
                   , 'PRI_CURRENT_BALANCE', 'PRI_SANCTIONED_AMOUNT', 'PRI_DISBURSED_AMOUNT', 'PRIMARY_INSTAL_AMT']
    dataset.loc[no_bureau, bureau_vars] = np.nan

    ## The 'Credit Score' variable PERFORM_CNS_SCORE has some issues and could use some additional feature engineering.
    ## The values of 300, 738, and 825 are over-represented in the data (300 should be at the end of the distribution)
    ## The values 11,14-18 are clearly 'Not Scored' codes - setting to missing for demo
    # dataset.PERFORM_CNS_SCORE.value_counts()
    # dataset.PERFORM_CNS_SCORE_DESCRIPTION.value_counts().sort_index()
    # pd.crosstab(dataset.PERFORM_CNS_SCORE_DESCRIPTION, dataset.PERFORM_CNS_SCORE, margins=True)
    dataset.loc[dataset.PERFORM_CNS_SCORE < 20, 'PERFORM_CNS_SCORE'] = np.nan

    ## Make all date calculation relative to January 2019 when this dataset was created.
    t_0 = pd.to_datetime('201901', format='%Y%m')
    dataset['DoB'] = pd.to_datetime(dataset['Date_of_Birth'], format='%d-%m-%y', errors='coerce')
    dataset['DoB'] = dataset['DoB'].mask( dataset['DoB'].dt.year > t_0.year
                                        , dataset['DoB'] - pd.offsets.DateOffset(years=100))
    dataset['AgeInMonths'] = (t_0 - dataset.DoB).astype('timedelta64[M]')

    dataset['DaysSinceDisbursement'] = (t_0 - pd.to_datetime(dataset.DisbursalDate, format='%d-%m-%y')
                                       ).astype('timedelta64[D]')

    def timestr_to_mths(timestr):
        '''timestr formatted as 'Xyrs Ymon' '''
        year = int(timestr.split()[0].split('y')[0]) 
        mo = int(timestr.split()[1].split('m')[0])
        num = year*12 + mo
        return(num)

    dataset['AcctAgeInMonths'] = dataset['AVERAGE_ACCT_AGE'].apply(lambda x: timestr_to_mths(x))
    dataset['CreditHistLenInMonths'] = dataset["CREDIT_HISTORY_LENGTH"].apply(lambda x: timestr_to_mths(x))

    dat = dataset.drop(columns=['Date_of_Birth', 'DoB', 'AVERAGE_ACCT_AGE', 'CREDIT_HISTORY_LENGTH'
                          , 'MobileNo_Avl_Flag', 'DisbursalDate'] )
    dat[targ_var] = dat[targ_var].astype(int)

    # ## Can drop records with no credit history - just to trim the data (justifiable in scenarios where
    # ## no_credit_bureau leads to an auto-decline or initiates a separate adjudication process)
    # dat = dat.loc[(~no_bureau | (dat.SEC_NO_OF_ACCTS != 0)), :]
    
    ## Drop some variables that are not good for scorecarding (sparse, high cardinality)
    ## The variable 'branch_id' is likely linked to geography and therefore demographics
    dat = dat.drop(columns=['supplier_id', 'Current_pincode_ID', 'Employee_code_ID', 'branch_id'])
    
    ## Give some variables shorter names 
    dat.rename(columns={'PERFORM_CNS_SCORE_DESCRIPTION': 'PERF_CNS_SC_DESC'
                   , 'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS': 'DELI_ACCTS_LAST_6_MTHS'
                   , 'NEW_ACCTS_IN_LAST_SIX_MONTHS': 'NEW_ACCTS_LAST_6_MTHS'}, inplace=True)
    
    return dat