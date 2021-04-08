#!/usr/bin/env python
# coding: utf-8

# ## Utility functions for weight of evidence scorecarding

import cudf
import cuml
import cupy

import numpy as np
import pandas as pd
import dask
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors

from scipy.special import logit, expit

# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

plt.style.use('default')
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['grid.color'] = '.8'


def preprocess_vehicle_data(dataset, id_vars, targ_var):
    ## Sort data by unique client identifier
    dataset = dataset.sort_values(id_vars)
    dataset = dataset.reset_index(drop=True)

    ## Make the target variable the second column in the dataframe
    targets = dataset.pop(targ_var)
    dataset.insert(1, targ_var, targets)

    ## Replace periods in variable names with underscores 
    new_cols = [sub.replace('.', '_') for sub in dataset.columns] 
    dataset.rename( columns=dict(zip(dataset.columns, new_cols)), inplace=True)

    ## Specify variables that should be treated as categorical and convert them to character strings (non-numeric)
    #cat_vars = [ 'branch_id', 'supplier_id', 'manufacturer_id', 'Current_pincode_ID', 'State_ID', 'Employee_code_ID'
    #            , 'Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag', 'Passport_flag']
    #dataset[cat_vars] = dataset[cat_vars].fillna(0)
    #dataset[cat_vars] = dataset[cat_vars].applymap(str)

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
    dataset['DoB'] = cudf.to_datetime(dataset['Date_of_Birth'], format='%d-%m-%y', errors='coerce')
    #dataset['DoB'] = dataset['DoB'].mask( dataset['DoB'].dt.year > t_0.year
    #                                    , dataset['DoB'] - cudf.offsets.DateOffset(years=100))
    #dataset['AgeInMonths'] = (t_0 - dataset.DoB).astype('timedelta64[M]')

    #dataset['DaysSinceDisbursement'] = (t_0 - cudf.to_datetime(dataset.DisbursalDate, format='%d-%m-%y')
    #                                   ).astype('timedelta64[D]')

    def timestr_to_mths(timestr):
        '''timestr formatted as 'Xyrs Ymon' '''
        year = int(timestr.split()[0].split('y')[0]) 
        mo = int(timestr.split()[1].split('m')[0])
        num = year*12 + mo
        return(num)

    #dataset['AcctAgeInMonths'] = dataset['AVERAGE_ACCT_AGE'].apply(lambda x: timestr_to_mths(x))
    #dataset['CreditHistLenInMonths'] = dataset["CREDIT_HISTORY_LENGTH"].apply(lambda x: timestr_to_mths(x))

    dat = dataset.drop(columns=['Date_of_Birth', 'DoB', 'AVERAGE_ACCT_AGE', 'CREDIT_HISTORY_LENGTH'
                          , 'MobileNo_Avl_Flag', 'DisbursalDate'] )
    dat[targ_var] = dat[targ_var].astype(int)

    # ## Can drop records with no credit history - just to trim the data (justifiable in scenarios where
    # ## no_credit_bureau leads to an auto-decline or initiates a separate adjudication process)
    # dat = dat.loc[(~no_bureau | (dat.SEC_NO_OF_ACCTS != 0)), :]
    
    return dat


def inv_logit(x, logbase=None):
    """Inverse logit function (expit from scipy equivalent to logbase=np.e)"""
    if not logbase:
        return expit(x)
    else:
        return 1.0 / ( 1.0 + logbase**(-x) )


def describe_data_g_targ(dat, target_var, logbase=cupy.e):
    """ """
    num = dat.shape[0]
    n_targ = (dat[target_var]==1).sum()
    n_ctrl = (dat[target_var]==0).sum()
    assert n_ctrl + n_targ == num
    base_rate = n_targ/num
    base_odds = n_targ/n_ctrl
    lbm = 1/cupy.log(logbase)
    base_log_odds = cupy.log(base_odds) * lbm    
    NLL_null = -(cupy.asarray(dat[target_var]) * cupy.log(base_rate) * lbm + cupy.asarray(1-dat[target_var]) * 
                 cupy.log(1-base_rate)*lbm).sum()
    LogLoss_null = NLL_null/num
    min_bin_ct = cupy.ceil(1/base_rate).astype(int)
    
    print("Number of records (num):", num)
    print("Target count (n_targ):", n_targ)
    print("Target rate (base_rate):", base_rate)
    print("Target odds (base_odds):", base_odds)
    print("Target log odds (base_log_odds):", base_log_odds)
    print("Dummy model negative log-likelihood (NLL_null):", NLL_null)
    print("Dummy model LogLoss (LogLoss_null):", LogLoss_null)
    # print("Minimum suggested bin size:", min_bin_ct)
    print("")
    return {'num':num, 'n_targ':n_targ , 'base_rate':base_rate, 'base_odds':base_odds
           , 'base_log_odds':base_log_odds, 'NLL_null':NLL_null, 'LogLoss_null':LogLoss_null 
           , 'min_bin_ct':min_bin_ct }


def get_numeric_cutpoints( notmiss_df, n_cuts=2, binner='qntl', min_bin_size=100 ):
    """Generate n cutpoints using specified binner.
    
    Parameters
    ----------
    notmiss_df : DataFrame
        A dataframe that contains both the target (column 0) and predictor variable (column 1). The dataframe
        should contain no missing values.
    n_cuts : int, optional (default=2)
        The number of data splits (generally creates n_cuts+1 bins)
    binner : string, optional (default='qntl')
        The spliting algorthm used
        - If 'qntl', then quantile binning is performed 
        - If 'gini', then Gini impurity criterion is used for recursive partitioning in decision tree
        - If 'entropy', then information gain criterion for split quality is used 
    min_bin_size : int, optional (default=100)
        Minimum count required in each bin generated by the 'binner' algorithm
    
    Returns
    -------
    var_cuts_w_min : Series 
        Series of cut points that includes the variable minimum as its first element.
    binner : str
        The binning algorithm used; binner will become 'qntl' for any undefined input binner string
    """
    targ_var, p_var, *_ =  notmiss_df.columns
    
    if (binner == 'gini' or binner =='entropy') and (n_cuts > 1):
        # dtree_ = tree.DecisionTreeClassifier(criterion=binner, max_leaf_nodes=n_cuts, min_samples_leaf=min_bin_size )
        dtree = RandomForestClassifier(n_estimators=1, criterion=binner, max_leaf_nodes=n_cuts, bootstrap=False
                               , min_samples_leaf=min_bin_size, max_features=None )
        dtree.fit(notmiss_df[[p_var]], notmiss_df[targ_var])
        tree0 = dtree.estimators_[0].tree_  # = dtree.tree_ ## For std. decision tree classifier
        leafmask = (tree0.feature != -2)
        prog_cuts = pd.Series(tree0.threshold[leafmask], index=tree0.children_left[leafmask]).sort_index()
        var_cuts_w_min = pd.concat([pd.Series(notmiss_df[p_var].min()), prog_cuts])
    elif (binner == 'unif'):
        bin_edges = np.linspace(notmiss_df[p_var].min(), notmiss_df[p_var].max(), n_cuts+1)
        var_cuts_w_min = pd.Series(bin_edges[:-1])
    elif (binner == 'hist'):  ## hist is the same as 'unif', but slightly slower
        _, bin_edges = np.histogram(notmiss_df[p_var], bins=n_cuts)
        var_cuts_w_min = pd.Series(bin_edges[:-1])
    else:  ## Perform quantile binning (the default)
        binner = 'qntl'
        var_cuts_w_min = notmiss_df[p_var].quantile(np.arange(0, 1, 1/np.where(n_cuts<1, 1, n_cuts)))       
        
    return(var_cuts_w_min.round(12), binner)


def get_categories( notmiss_df, n_cats=2, binner=None, min_bin_size=100 ):
    """Generate n categories using specified binner
    
    Parameters
    ----------
    notmiss_df : DataFrame
        A dataframe that contains both the target (column 0) and predictor variable (column 1). The dataframe
        should contain no missing values.
    n_cats : int, optional (default=2)
        The number of data categories to return (generally creates n_cats+1 bins)
    binner : string, optional (default=None)
        The spliting algorthm used
        - If 'gini', then Gini impurity criterion is used for recursive partitioning in decision tree
        - If 'entropy', then information gain criterion for split quality is used 
        - Otherwise, categories are rank ordered by size and the first 'n_cats' categories are returned
    min_bin_size : int, optional (default=100)
        Minimum count required in each bin generated by the 'binner' algorithm
    
    Returns
    -------
    incl_cats : list 
        A list of category item strings
    binner : str
        The binning algorithm used. Returned to use as a flag ('none' is a stopping condition for scorecard growth)
    """
    targ_var, p_var, *_ =  notmiss_df.columns
    
    var_cts = notmiss_df[p_var].value_counts()
    small_cats = var_cts[var_cts < min_bin_size].index.to_arrow().to_pylist()   #.tolist()                                           
    low_rank_cats = var_cts[(n_cats-1):].index.to_arrow().to_pylist()           #.tolist()                                           
    
    if (len(low_rank_cats) <= len(small_cats)) or (len(low_rank_cats) <= 1):
        binner = 'none' 
        other_set = set(low_rank_cats + small_cats)
        if len(other_set) <= 1:
            incl_cats = var_cts.index.to_arrow().to_pylist()                    #.tolist()                                           
        else: 
            incl_cats = list(set(var_cts.index) - other_set)
    elif (binner == 'gini' or binner =='entropy') and (n_cats > 1):
        # dtree_ = tree.DecisionTreeClassifier(criterion=binner, max_leaf_nodes=n_cats, min_samples_leaf=min_bin_size )
        dtree = RandomForestClassifier(n_estimators=1, criterion=binner, max_leaf_nodes=n_cats, bootstrap=False
                               , min_samples_leaf=min_bin_size, max_features=None )
        dxx = pd.get_dummies(notmiss_df[p_var])
        dxx = dxx.loc[:, ~dxx.columns.isin(small_cats)]
        dtree.fit(dxx, notmiss_df[targ_var])
        feats = dtree.estimators_[0].tree_.feature  # = dtree.tree_.feature ## For std. dec. tree
        incl_cats = dxx.columns[feats[feats != -2]].tolist()
    else:
        binner = 'rank'
        other_set = set(low_rank_cats + small_cats)
        incl_cats = list(set(var_cts.index) - other_set)

    return(incl_cats, binner)


def get_bin_edges( dat_df, n_cuts, binner='qntl', min_bin_size=100, correct_cardinality=True ):
    """Function to generate bins from a 'binner' algorithm 
    
    Parameters
    ----------
    dat_df : DataFrame
        A dataframe that contains both the target (column 0) and predictor variable (column 1) 
    n_cuts : int, optional (default=2)
        The number of data splits (generally creates n_cuts+1 bins)
    binner : string, optional (default='qntl')
        The spliting algorthm used
        - If 'qntl', then quantile binning is performed 
        - If 'gini', then Gini impurity criterion is used for recursive partitioning in decision tree
        - If 'entropy', then information gain criterion for split quality is used 
    min_bin_size : int, optional (default=100)
        Minimum count required in each bin generated by the 'binner' algorithm
    correct_cardinality: bool, optional (default=True)
         Toggle that attempts to correct for the fact that when min_bin_size > 1 the effective cardinality 
         when binning numeric variables is lower than the actual cardinality.
    
    Returns
    -------
    bin_edges : Series
        A series of bin edges (either cut-points or category sets).  
    binner : string
        The binning algorithm used. Returned to use as a flag ('none' is a stopping condition for scorecard growth)
    """
    targ_var, p_var, *_ = dat_df.columns

    ## Determine if the data is numeric
    ## np.issubtype will return an error if given a pd.categorical variable, hence the pre-condition.
    is_numeric = (not hasattr(dat_df[p_var], 'cat') and cupy.issubdtype(dat_df[p_var], cupy.number))

    if is_numeric:
        srtd_var_cts = dat_df[p_var].value_counts(sort=False).sort_index()
        desc_var_cts = srtd_var_cts.iloc[::-1]  # srtd_var_cts.sort_index(ascending=False) ## Works, but slower

        cardinality0 = len(srtd_var_cts)
        if correct_cardinality:
            ## Correcting cardinality by eliminating unique values in both tails of the distribution until
            ## the minimum bin size is reached cumulatively (only really useful for count variables)
            cum_desc_var_cts = desc_var_cts.cumsum().sort_index()          
            cardinality = cudf.DataFrame(([((srtd_var_cts.cumsum() >= min_bin_size) & 
                                            (cum_desc_var_cts >= min_bin_size)).sum(),
                                            (cum_desc_var_cts//min_bin_size).nunique()-1 ])).min()[0]
        else:
            cardinality = cardinality0

    ## Get initial indicator that predictor variable data is missing (missing categories 
    missing_ind = dat_df[p_var].isnull() 

    if (is_numeric and (cardinality > n_cuts)):
        notmiss_df = dat_df.loc[~missing_ind, [targ_var, p_var]] 
        bin_edges, binner = get_numeric_cutpoints(notmiss_df, n_cuts, binner, min_bin_size) 
    elif (is_numeric and (cardinality0 > 2)): 
        ## If cardinality is less than n_cuts then treat as an ordinal categorical variable - count variables
        binner = 'none'
        inc_bins = (srtd_var_cts >= min_bin_size)

        ## Include bins that have more than minimum bin size - summing from end of right tail
        cnt = 0 
        for index, value in desc_var_cts.items():
            cnt += value
            if cnt >= min_bin_size:
                inc_bins[index] = True 
                cnt = 0

        ## If minimum value not in included bins, then add it after removing the current first included bin
        if ~inc_bins.iloc[0]:
            inc_bins.loc[inc_bins.idxmax()] = False
            inc_bins.iloc[0] = True    
        bin_edges = srtd_var_cts[inc_bins].index.to_series().sort_values()
    else:
        dat_df[p_var] = dat_df[p_var].astype(str).str.strip()
        missing_ind = (missing_ind | (dat_df[p_var] == ''))
        notmiss_df = dat_df.loc[~missing_ind, [targ_var, p_var]] 

        cat_sets, binner = get_categories(notmiss_df, n_cuts, binner, min_bin_size)
        bin_edges = cudf.Series(cat_sets)
    
    return(bin_edges, binner)



def gen_uwoesc_df( dat_df, bin_edges, binner=None, n_cuts=None, min_bin_size=100, laplace=1, laplaceY0='brc'
                  , compute_stats=False, neutralizeMissing=False, neutralizeSmBins=True, logbase=np.e ):
    """Function that generates a univariate WOE dataframe given a dataset and bin specification.
    
    Parameters
    ----------
    dat_df : DataFrame
        A dataframe that contains both the target (column 0) and predictor variable (column 1) 
    bin_edges : list, Series
        A list or series of numeric cut points or category sets. Categories can be specified as strings or sets.
    binner : string, optional (default=None)
        The spliting algorthm used to generate the bin edges, passed from gen_bin_edges() function to included in
        final WOE dataframe (i.e., not functionally effective)
    n_cuts : int, optional (default=2)
        The number of data splits. Just passed to insert into final WOE dataframe.
    min_bin_size : int, optional (default=100)
        Minimum bin size allowed. If created, bins with fewer than min_bin_size instances will have WOE = 0
        if neutralizeSmBins=True.
    laplace : int, float, optional (default=1)
        Additive (Laplace) smoothing parameter used when estimating target (Y=1) distribution probabilities
    laplaceY0 : int, float, string, optional (default='brc')
        Additive (Laplace) smoothing parameter used when estimating the non-target/control (Y=0) distribution
        - If int or float, then value is used for additive smoothing
        - If 'bal' or balanced, then laplace parameter is used
        - If 'brc' or base rate corrected, then laplace parameter divided by base_odds is used. Makes additive
            smoothing respect the prior/base rate of the target in the dataset.
    logbase: int, float, optional (default=np.e) 
        The base for logarithm functions used to compute log-odds - default is natural log (ln = log_e)
    compute_stats: bool, optional (default=False)
        Toggle for computing stats - KL divergences (relative entropies), IVs, bin prediction, etc.
    neutralizeMissing: bool, optional (default=False)
        Toggle to set WOE values for missing value bins equal to zero
    neutralizeSmBins: bool, optional (default=True)
        Toggle that sets WOE values to zero when the bin count is less than min_bin_size. Generating bins with 
        counts less than min_bin_size is uncommon, but does occur (e.g., missing bins, quantile bins next to bins 
        with numerous ties, etc.)
    
    Returns
    -------
    WOE_df : DataFrame 
        A univariate scorecard stored as a pandas dataframe
    """
    targ_var, p_var, *_ = dat_df.columns
    base_odds = ((dat_df[targ_var]==1) / (dat_df[targ_var]==0).sum()).sum()

    if not binner:
        binner = 'custom'
        n_cuts = -1 # len(bin_edges)

    missing_ind = dat_df[p_var].isnull() 

    bin_edges = cudf.DataFrame(bin_edges) # GPU
    
    if len(bin_edges)==0 or isinstance(list(bin_edges)[0], str) or isinstance(list(bin_edges)[0], set):
        var_type = 'C'
        dat_df[p_var] = dat_df[p_var].astype(str).str.strip()
        missing_ind = (missing_ind | (dat_df[p_var] == ''))
    else:
        var_type = 'N'

    notmiss_df = dat_df.loc[~missing_ind,:].copy()
    missing_df = dat_df.loc[missing_ind, :]

    ## Get the log base multiplier for log base conversion
    lbm = 1.0 / np.log(logbase) 
    #lbm = 1.0 / cupy.log(logbase)                                                                                               

    WOE_df = cudf.DataFrame()   
    if var_type == 'C':
        missing_value = {''}
        input_cats = set()
        cat2set_dict = {}
        for cat_set in bin_edges:
            if isinstance(cat_set, str):
                cat_set = {cat_set}
            input_cats = input_cats.union(cat_set)
            cat2set_dict.update( dict.fromkeys( cat_set, cat_set ) )
        other_cats = set(notmiss_df[p_var].unique()) - input_cats

        if len(other_cats) > 1:
            cat2set_dict.update( dict.fromkeys(other_cats, {'Other'}) )

        map_dict = {key: ','.join(value) for key, value in cat2set_dict.items()}    

        notmiss_df['bin'] = notmiss_df[p_var].map( map_dict ).astype(str)
        dfgb = notmiss_df.groupby('bin', as_index=True)
        bin_ct = dfgb[targ_var].count()
        if len(other_cats) <= 1 and missing_ind.sum() > 0:  # binner == 'none' and
            bin_ct['Other'] = 0         
        WOE_df['bin_ct'] = bin_ct
        WOE_df['bin_min'] = WOE_df.index.astype(str).to_series().str.split(',').apply(set)
        WOE_df['ranks'] = WOE_df.bin_ct.where(WOE_df.bin_min != {'Other'}).rank(ascending=False)
        WOE_df.loc[WOE_df.bin_min == {'Other'}, 'ranks'] = WOE_df.shape[0]
    else:
        missing_value = cupy.nan
        if isinstance(bin_edges, list):
            min_val = notmiss_df[p_var].min()
            if bin_edges[0] > min_val:
                print("Adding minimum value to cut string.")
                bin_edges = [min_val] + bin_edges
            bin_edges = pd.Series(bin_edges)

        srtd_bin_edges = bin_edges.sort_values(by=[0]) # GPU
        #var_cuts_w_range = np.append(srtd_bin_edges.drop_duplicates().values, np.inf)
        var_cuts_w_range = cudf.concat([srtd_bin_edges.drop_duplicates(), 
                                        cudf.DataFrame([cupy.inf])]).values.flatten() # GPU
        #notmiss_df['bin'] = pd.cut(notmiss_df[p_var], var_cuts_w_range, right=False)
        notmiss_df['bin'] = cupy.digitize(cupy.asarray(notmiss_df[p_var]), cupy.asarray(var_cuts_w_range))        
        dfgb = notmiss_df.groupby('bin', as_index=True, sort=True)
        WOE_df['bin_ct'] = dfgb[targ_var].count()        
        WOE_df['bin_min'] = var_cuts_w_range[:-1]  ## Drop max value 
        ranks_min = srtd_bin_edges.rank(method='min').drop_duplicates().values
        ranks_max = srtd_bin_edges.rank(method='max').drop_duplicates().values
        WOE_df['ranks'] = [(x1, x2) for x1,x2 in zip(ranks_min, ranks_max)]

    WOE_df['Y1'] = dfgb[targ_var].sum() # GPU
    WOE_df['Y1'] = WOE_df['Y1'].fillna(0).astype(int)  
    #WOE_df.sort_values(by=['ranks','Y1'], ascending=[True,False], inplace=True)
    WOE_df = WOE_df.sort_values(by=['ranks','Y1'], ascending=[True,True]) # GPU
    #missing_row = cudf.DataFrame([[missing_df[targ_var].count(), missing_value, 0, missing_df[targ_var].sum()]]
    #                           , columns=WOE_df.columns, index=cudf.Index(['.'], name='bin'))       
    missing_row = cudf.DataFrame([[missing_df[targ_var].count(), missing_value, 0, missing_df[targ_var].sum()]], 
                                 index=cudf.Index([0], name='bin'), 
                                 columns=WOE_df.columns.values.tolist() )  
    #WOE_df = pd.concat([missing_row, WOE_df])
    WOE_df = cudf.concat([missing_row.astype('float32'), WOE_df.astype('float32')])
    WOE_df['Y0'] = WOE_df.bin_ct - WOE_df.Y1
    WOE_df['bin_pct'] = WOE_df.bin_ct / WOE_df.bin_ct.sum()
    #WOE_df['targ_rate'] = WOE_df.Y1 / cupy.where(WOE_df.bin_ct==0, 1, WOE_df.bin_ct)
    WOE_df['targ_rate'] = cupy.asarray(WOE_df.Y1) / cupy.where(WOE_df.bin_ct==0, 1, WOE_df.bin_ct)  ## value of 1 is arbitrary 

    if laplaceY0 == 'bal':
        laplaceY0 = laplace
        base_odds = 1 
    elif laplaceY0 == 'brc':
        laplaceY0 = laplace/base_odds

    if laplace <= 0:
        pr_rate = 1.0 / (1.0 + np.exp(-np.log(base_odds)))
        null_event_fill_Y1 = pr_rate  ## Can set this to another value
        null_event_fill_Y0 = null_event_fill_Y1 / base_odds 
        Y1_ct = np.maximum(WOE_df.Y1, null_event_fill_Y1)
        Y0_ct = np.maximum(WOE_df.Y0, null_event_fill_Y0)
    else:
        Y1_ct = WOE_df.Y1 + laplace
        Y0_ct = WOE_df.Y0 + laplaceY0 

    WOE_df['p_XgY1'] = Y1_ct / Y1_ct.sum()
    WOE_df['p_XgY0'] = Y0_ct / Y0_ct.sum() 
    #WOE_df['WOE'] = cupy.log(WOE_df.p_XgY1 / WOE_df.p_XgY0).fillna(0)*lbm    
    WOE_df['WOE'] = cudf.Series(cupy.log(WOE_df.p_XgY1 / WOE_df.p_XgY0)).fillna(0)*lbm  
    
    if neutralizeSmBins: 
        WOE_df.loc[WOE_df.bin_ct < min_bin_size, 'WOE'] = 0.0

    if neutralizeMissing:
        WOE_df.loc[(WOE_df.bin_min.isna() | (WOE_df.bin_min == missing_value)), 'WOE'] = 0.0

    bins_out = WOE_df.shape[0]
    WOE_df['var_name'] = p_var
    WOE_df['binner'] = binner
    WOE_df['req_cuts'] = n_cuts
    WOE_df['bins'] = bins_out
    WOE_df['bin_idx'] = list(range(bins_out))

    WOE_df.index = WOE_df.index.astype(str)
    WOE_df.reset_index(inplace=True)
    WOE_df.set_index(['var_name','binner','req_cuts','bins','bin_idx'], inplace=True)

    if compute_stats:
        WOE_df['KLY1Y0'] = WOE_df.p_XgY1 * WOE_df.WOE 
        WOE_df['KLY0Y1'] = WOE_df.p_XgY0 * -WOE_df.WOE
        # WOE_df['IV'] = (WOE_df.p_XgY1-WOE_df.p_XgY0) * WOE_df.WOE
        WOE_df['IV'] = WOE_df['KLY1Y0'] + WOE_df['KLY0Y1']
        WOE_df['nIV'] = WOE_df['IV'] / np.log(bins_out)*lbm
        WOE_df['bin_pred'] = 1.0 / (1.0 + logbase**(-(np.log(base_odds)*lbm + WOE_df.WOE)))

    return(WOE_df)


def gen_woe_df( dat_df, p_var, targ_var, n_cuts=2, laplace=1, min_bin_size=100, binner='qntl'
                , laplaceY0='brc', compute_stats=False, neutralizeMissing=False, neutralizeSmBins=True
                , correct_cardinality=True, logbase=np.e ):
    """Generate a Weight-of-Evidence (WOE) dataframe using a binning algorithm (binner)
    
    Parameters
    ----------
    dat_df : DataFrame
        A dataframe that contains both the predictor and target variable
    p_var : string
        The predictor variable.
    targ_var : string
        The model target variable, corresponding to a binary variable in dat.
    n_cuts : int, optional (default=2)
        The number of data splits (generally creates n_cuts+1 bins)
    laplace : int, float, optional (default=1)
        Additive (Laplace) smoothing parameter used when estimating target (Y=1) distribution probabilities
    min_bin_size : int, optional (default=100)
        Minimum bin size allowed. If created, bins with fewer than min_bin_size instances will have WOE = 0
        if neutralizeSmBins=True.
    binner : string, optional (default='qntl')
        The spliting algorthm used
        - If 'qntl', then quantile binning is performed 
        - If 'gini', then Gini impurity criterion is used for recursive partitioning in decision tree
        - If 'entropy', then information gain criterion for split quality is used 
    laplaceY0 : int, float, string, optional (default='brc')
        Additive (Laplace) smoothing parameter used when estimating the non-target/control (Y=0) distribution
        - If int or float, then value is used for additive smoothing
        - If 'bal' or balanced, then laplace parameter is used
        - If 'brc' or base rate corrected, then laplace parameter divided by base_odds is used. Makes additive
            smoothing respect the prior/base rate of the target in the dataset.
    compute_stats: bool, optional (default=False)
        Toggle for computing stats - KL divergences (relative entropies), IVs, bin prediction, etc.
    neutralizeMissing: bool, optional (default=False)
        Toggle to set WOE values for missing value bins equal to zero
    neutralizeSmBins: bool, optional (default=True)
        Toggle that sets WOE values to zero when the bin count is less than min_bin_size. Generating bins with 
        counts less than min_bin_size is uncommon, but does occur (e.g., missing bins, quantile bins next to bins 
        with numerous ties, etc.)
    correct_cardinality: bool, optional (default=True)
         Toggle that attempts to correct for the fact that when min_bin_size > 1 the effective cardinality 
         when binning numeric variables is lower than the actual cardinality.
    logbase: int, float, optional (default=np.e) 
        The base for logarithm functions used to compute log-odds - default is natural log (ln = log_e)
    
    Returns
    -------
    WOE_df : DataFrame  
        A univariate scorecard stored as a pandas dataframe
    """
    dat_df = dat_df.loc[:,[targ_var, p_var]]
    bin_edges, binner = get_bin_edges(dat_df, n_cuts, binner, min_bin_size, correct_cardinality)
    WOE_df = gen_uwoesc_df( dat_df, bin_edges, binner, n_cuts, min_bin_size, laplace, laplaceY0
                          , compute_stats, neutralizeMissing, neutralizeSmBins, logbase )
    return(WOE_df)


# In[ ]:


def woe_score_var( dat, WOE_df, return_only_WOE=True ):
    """Perform weight of evidence transformation on a single predictor variable
    
    Parameters
    ----------
    dat : DataFrame
        A dataframe that WOE transformed variable will concatenated to
    WOE_df : DataFrame
        A dataframe that contains a univariate WOE dataframe model object.
    return_only_WOE: boolean, optional (default=True)
        Toggle to return only the WOE transformed data. If False, dat is returned with WOE transformed variable 
        attached.
    
    Returns
    -------
    wdat : DataFrame
        A dataframe including the WOE transformed predictor variable   
    """
    wdat = dat.copy()
    
    p_var = WOE_df.index.get_level_values('var_name')[0]
    
    if not isinstance(WOE_df.bin_min[1], set):     
        cutpoints = list(WOE_df.bin_min.dropna()) + [np.inf]
        if wdat[p_var].min() < cutpoints[0]:
            cutpoints[0] = wdat[p_var].min()
        wdat['bin_idx'] = (pd.cut(wdat[p_var], bins=cutpoints, labels=False, right=False)+1).fillna(0).astype(int)
    else: 
        wdat[p_var].fillna('', inplace=True)
        wdat[p_var] = wdat[p_var].astype(str).str.strip().astype('category')
        wdat_cats0 = wdat[p_var].cat.categories  # .astype(str)

        cat_sets = WOE_df.bin_min.tolist() # [:-1]
        sets_series = pd.Series(cat_sets).astype(str)
        sc_cats = pd.Series(sorted(set().union(*cat_sets)))

        cat2set_dict = {}
        for cat_set in cat_sets:
            cat2set_dict.update( dict.fromkeys( cat_set, cat_set ) )

        if any(~wdat_cats0.isin(sc_cats)):
            if 'Other' in sc_cats.values:
                print(p_var+" - Creating 'Other' category from:", wdat_cats0[~wdat_cats0.isin(sc_cats)].values)
                cat_dict = dict.fromkeys(wdat_cats0[~wdat_cats0.isin(sc_cats)], {'Other'}) 
            else:
                print(p_var+" - Values added to missing:", wdat_cats0[~wdat_cats0.isin(sc_cats)].values)
                cat_dict = dict.fromkeys(wdat_cats0[~wdat_cats0.isin(sc_cats)], {''})
            cat2set_dict.update( cat_dict )

        wdat[p_var] = wdat[p_var].map(cat2set_dict).astype(str).astype('category')
        wdat_cats1 = wdat[p_var].cat.categories

        if any(~sets_series.isin(wdat_cats1)):   
            wdat[p_var].cat.add_categories(sets_series[~sets_series.isin(wdat_cats1)].values, inplace=True)

        wdat[p_var].cat.reorder_categories(sets_series, inplace=True)
        wdat['bin_idx'] = wdat[p_var].cat.codes
        
    bin2WOE = WOE_df[['WOE']].rename(columns={'WOE': 'WOE_'+p_var}).reset_index(
                level=['var_name','req_cuts','binner','bins'], drop=True)
    wdat = wdat.reset_index().merge(bin2WOE, how='left', on='bin_idx').set_index('index')

    if return_only_WOE:
        return(wdat[['WOE_'+p_var]])
    else:
        return(wdat)



## The first function here can be made more efficient by reusing cutpoints for tree binners...

def gen_var_woe_dfs(p_var, inc_n_cuts, binner, dat, targ_var, **kwargs ):
    """Helper function run in each dask instance
    """
    WOE_dfs = pd.DataFrame()
    for wdf_idx, n_cuts in enumerate(sorted(inc_n_cuts)):
        print(p_var, binner, n_cuts)
        if wdf_idx > 0 and WOE_df.index.get_level_values('binner')[0] == 'none':
            break
        WOE_df = gen_woe_df( dat, p_var, targ_var, n_cuts, binner=binner, **kwargs )
        WOE_dfs = WOE_dfs.append(WOE_df, sort=False)
    return(WOE_dfs)

def create_woesc_df( inc_p_vars, inc_n_cuts, inc_bnrs, dat_df, targ_var, **kwargs ):
    """Create a set of WOE scorecards
    
    Parameters
    ----------
    inc_p_vars : list
        A list of predictor variables for which univariate scorecards will be made
    inc_n_cuts : list
        A list of cut counts - specifying how many cuts to make
    inc_bnrs : list
        A list of binners, e.g. ['qntl', 'entropy', 'gini']
    dat_df : DataFrame
        A dataframe that contains the target variable and all predictor variables in inc_p_vars 
    targ_var : string
        The model target variable, corresponding to a binary variable in dat.
    ... additional keyword arguments will be passed through to gen_woe_df()
        
    Returns
    -------
    mdf : DataFrame
        A dataframe containing the outer-product of all the univariate scorecards requested
    """
    delayed_res = []
    for p_var in inc_p_vars:
        mdat = dat_df.loc[:,[targ_var, p_var]] # .copy()
        for bnr in inc_bnrs:
            dres = dask.delayed(gen_var_woe_dfs, pure=True)( 
                                    p_var, inc_n_cuts, bnr, mdat, targ_var, **kwargs )
            delayed_res.append(dres)

    computed_set = dask.compute(*delayed_res, scheduler='processes', num_workers=24)
    # computed_set = dask.compute(*delayed_res, scheduler='single-threaded')

    mdf = pd.concat(computed_set)
    mdf.sort_index(inplace=True)
    mdf = mdf[~mdf.index.duplicated(keep='first')]
    
    mdf['KLY1Y0'] = mdf.p_XgY1*mdf.WOE 
    mdf['KLY0Y1'] = mdf.p_XgY0*-mdf.WOE
    mdf['IV'] = mdf['KLY1Y0'] + mdf['KLY0Y1']
    
    base_odds = sum(dat_df[targ_var]==1)/sum(dat_df[targ_var]==0)
    mdf['bin_pred'] = inv_logit(np.log(base_odds) + mdf.WOE)
    return(mdf)


# In[ ]:


def gen_uscm_df(mdf):
    """Generate a univariate scorecards metrics dataframe from a univariate scorecards dataframe
    
    Parameters
    ----------
    mdf : DataFrame
        A dataframe containing univariate woe scorecard models to be evaluated.
        
    Returns
    -------
    MM_df : DataFrame
        A dataframe containing one summary row for each univariate scorecard in input 'mdf'
    """
    MM_dfgb = mdf.groupby(['var_name', 'binner', 'req_cuts', 'bins'])
    MM_df = MM_dfgb.IV.agg(['sum']) 
    MM_df.rename(columns={'sum': 'IV'}, inplace=True)

    MM_df.reset_index(['bins'], inplace=True)

    MM_df['nIV'] = MM_df.IV / np.log(MM_df.bins)
    MM_df['KLY1Y0'] = MM_dfgb.KLY1Y0.sum().values 
    MM_df['KLY0Y1'] = MM_dfgb.KLY0Y1.sum().values 
    
    monotonic_incr = MM_dfgb.WOE.apply(lambda x: x[1:].is_monotonic_increasing)
    monotonic_decr = MM_dfgb.WOE.apply(lambda x: x[1:].is_monotonic_decreasing)
    MM_df['monotonic'] = (monotonic_incr | monotonic_decr).values.astype('object')
    MM_df['monotonicity'] = (monotonic_incr.astype(int) - monotonic_decr.astype(int)).values

    ind_cat = MM_dfgb.bin_min.first().apply(lambda x: isinstance(x, set)).values
    MM_df.loc[ind_cat, 'monotonic'] = None
    MM_df.loc[ind_cat, 'monotonicity'] = 0
    
    MM_df.sort_index(inplace=True)
    return(MM_df)


# In[ ]:


def uwoesc_plot( WOE_df, targ_label, sort_values=False, var_scale='def', top_n=None, sep_bar=False
                       , textsize=9, figsize=(8.2, 4)):
    """Create a plot of a univariate scorecard using a WOE_df object.
    """   
    
    WOE_df = WOE_df.to_pandas()   
    
    p_var = WOE_df.index[0][0]
    compute_stats = 'IV' in WOE_df.columns
    
    if top_n:
        WOE_df = WOE_df.iloc[:top_n+1,:]
    
    n_bins = WOE_df.index.get_level_values('bins')[0]
    
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    fig.tight_layout(pad=4.0)
    fig.text(0.5, 0.92, p_var, ha='center', size=20)
    xticks = np.arange(0, len(WOE_df.WOE))
    ax1_xticks = xticks
    axs[1].axhline(0, color='.8', lw=2.5)
    ave_bin_width = 1
    woeCol = 'purple'
    woeAlpha = .6
    
    if isinstance(WOE_df.bin_min[1], set):
        if sort_values:
            mask = (WOE_df.bin=='Other')
            WOE_df = pd.concat([WOE_df[~mask].sort_values('bin', na_position='first'), WOE_df[mask]])
        xlabel = "bin category"
        xticklabels = WOE_df.bin.tolist()
        ax1_xticklabels = xticklabels[1:]
        xtick_offset = 0 
        text_offset = -.4
        barwidth = .8  
        axs[1].bar(ax1_xticks[:-1], WOE_df.WOE[1:], width=barwidth, lw=1.5, fc=mpl_colors.to_rgb(woeCol)+(.3,)
                   , ec=mpl_colors.to_rgb(woeCol)+(woeAlpha,))
        axs[1].set_xticks(ax1_xticks[:-1])       
    else:
        if WOE_df.bin_min[1:].apply(float.is_integer).all():
            xticklabels = WOE_df.bin_min.map('{:.0f}'.format).tolist() 
        else:
            first_dec_digit = np.floor(np.log10(WOE_df.bin_min[1:].abs(), where=(WOE_df.bin_min[1:]!=0)))
            xtickdigits = np.min([3,np.nanmax([1,1-first_dec_digit.min().astype(int)])])
            xticklabels = WOE_df.bin_min.apply(lambda x:'{:.{}f}'.format(x, xtickdigits)).tolist() 
        xlabel = "bin cutpoints"
        ax1_xticklabels = xticklabels[1:] + ['max']
        xtick_offset = -.5
        text_offset = 0
        barwidth = 1
        if var_scale == 'orig' and len(WOE_df.bin_min) > 2:
            ave_bin_width = np.nanmin([(WOE_df.bin_min[2:] - WOE_df.bin_min[2:].shift()).mean()
                                   , (WOE_df.bin_min[-1] - WOE_df.bin_min[1])/(n_bins-1) ])
            x_init = np.max([WOE_df.bin_min[1], WOE_df.bin_min[2] - 2*ave_bin_width])
            ax1_xticks = np.array([x_init] + WOE_df.bin_min[2:].tolist() + [WOE_df.bin_min[-1] + 2*ave_bin_width])
            ax1_xticklabels[0] = 'min'
        axs[1].step(ax1_xticks, [WOE_df.WOE[1]] + WOE_df.WOE[1:].tolist(), color=woeCol, label='WOE', alpha=woeAlpha)
        axs[1].set_xticks(ax1_xticks) 
      
    ra = .42
    if sep_bar:
        axs[0].bar(xticks+.12, WOE_df['p_XgY1'], width=.6, label=targ_label+'=1', facecolor=(1,0,0,ra))
        axs[0].bar(xticks-.08, WOE_df['p_XgY0'], width=.6, label=targ_label+'=0', fc=(0,0,1,ra/(ra+1)))
    else:
        axs[0].bar(xticks, WOE_df['p_XgY1'], width=barwidth, label=targ_label+'=1', facecolor=(1,0,0,ra)) #, ec='w')
        axs[0].bar(xticks, WOE_df['p_XgY0'], width=barwidth, label=targ_label+'=0', fc='b', alpha=ra/(ra+1))
    axs[0].set_xticks(xticks+xtick_offset)
    axs[0].set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylim((0, np.where(axs[0].get_ylim()[1] < 2/(n_bins-1), 2/(n_bins-1), axs[0].get_ylim()[1])))
    axs[0].set_ylabel("Probability Density")
    axs[0].tick_params(labelsize=textsize)
    axs[0].legend(frameon=True, framealpha=1, fontsize=textsize)

    axs[1].axhline(WOE_df.WOE[0], linestyle=':', color=woeCol, alpha=woeAlpha)
    axs[1].set_xticklabels(ax1_xticklabels, rotation=45, ha='right', rotation_mode='anchor')
    axs[1].set_xlabel(xlabel)
    axs[1].set_xlim(left=axs[1].get_xlim()[0]-.7*ave_bin_width)
    axs[1].set_ylim( (np.where(axs[1].get_ylim()[0] > -1, -1, axs[1].get_ylim()[0])
                    , np.where(axs[1].get_ylim()[1] < 1, 1, axs[1].get_ylim()[1])) )
    axs[1].set_ylabel("WOE (est. marginal log-odds)")
    axs[1].tick_params(labelsize=textsize)
    
    if compute_stats:
        for i, x_pt in enumerate(ax1_xticks[:-1]):
            axs[1].text(x_pt+text_offset, WOE_df.WOE.values[i+1]+.01*np.diff(axs[1].get_ylim())
                , str(np.round((100*WOE_df.bin_pred.values)[i+1], 2)) +'%', color=woeCol, size=textsize-1)

        axs[1].text(axs[1].get_xlim()[0], WOE_df.WOE.values[0]+.01*np.diff(axs[1].get_ylim())
                , str(np.round((100*WOE_df.bin_pred.values)[0],2)) +'%', color=woeCol, size=textsize-1) 
        
        num = WOE_df.bin_ct.sum()
        n_targ = sum(WOE_df.Y1)
        MetricList = [ "# of Records = {:.0f}".format(num)
                        , "# of Targets = {:.0f}".format(n_targ)
                        , "# Missing = {:.0f}".format(WOE_df.iloc[0].bin_ct)
                        , "Base Rate = {:.3f}%".format(100*(n_targ/num))
                        , "RelEntropyY1Y0 = {:.5f}".format(sum(WOE_df.KLY1Y0))
                        , "RelEntropyY0Y1 = {:.5f}".format(sum(WOE_df.KLY0Y1))
                        , "InfoVal = {:.5f}".format(sum(WOE_df.IV))
                        , "nInfoVal = {:.5f}".format(sum(WOE_df.nIV))
                      ]
        ylocs = np.arange(.97,0,-.05)[:len(MetricList)]
        for yloc, sm in zip(ylocs, MetricList):
            axs[1].annotate(sm, xy=(1.02, yloc), xycoords='axes fraction', size=textsize)
    
    plt.show()
    return(WOE_df)

def univariate_sc_plot( dat_df, p_var, targ_var, n_cuts=2, laplace=1, min_bin_size=1, binner='qntl'
                       , compute_stats=True, sort_values=False, var_scale='def', top_n=None, sep_bar=False
                       , textsize=9, figsize=(8.2, 4), **kwargs):
    """Create a plot of a univariate scorecard.
    """
    WOE_df = gen_woe_df(dat_df, p_var, targ_var, n_cuts, laplace, min_bin_size, binner
                        , compute_stats=compute_stats, **kwargs)
    
    return uwoesc_plot( WOE_df, targ_var, sort_values=sort_values, var_scale=var_scale
                       , top_n=top_n, sep_bar=sep_bar, textsize=textsize, figsize=figsize)


# In[ ]:


def woesc_plot(WOE_df, compute_stats=True, sort_values=False, orig_scale=False, logbase=np.e, figsize=(5, 3)):
    """Make a weight of evidence plot.
    """   
    ## Re-scale values...
    WOE_df.loc[:, ['WOE','Score']] = WOE_df.loc[:, ['WOE','Score']]/np.log(logbase)
    
    p_var = WOE_df.index.get_level_values('var_name')[0]
    n_bins = WOE_df.index.get_level_values('bins')[0]

    fig, axs = plt.subplots(1, 1, figsize=figsize)
    axs.axhline(0, color='.7', lw=2.5)
    xticks = np.arange(0, len(WOE_df.WOE))
    ave_bin_width = 1 
    woeCol = 'purple' # '#7849B8'
    woeAlpha = .6
    
    if isinstance(WOE_df.bin_min[1], set):
        if sort_values:
            mask = (WOE_df.bin=='Other')
            WOE_df = pd.concat([WOE_df[~mask].sort_values('bin', na_position='first'), WOE_df[mask]])
        xlabel = "bin category"
        xticklabels = WOE_df.bin
        barwidth = .8
        xticks = xticks[:-1]
        axs.bar(xticks, WOE_df.WOE[1:], width=barwidth, ec=mpl_colors.to_rgb(woeCol)+(woeAlpha,), fill=False
                , label="WOE")
        axs.bar(xticks, WOE_df.Score[1:], width=barwidth, fc=(0, 0, 0, 0.2), ec='k', label="Score")
    else:            
        if WOE_df.bin_min[1:].apply(float.is_integer).all():
            xticklabels = WOE_df.bin_min.map('{:.0f}'.format).tolist() + ['max']
        else:
            first_dec_digit = np.floor(np.log10(WOE_df.bin_min[1:].abs().tolist(), where=(WOE_df.bin_min[1:]!=0)))
            xtickdigits = np.min([3,np.nanmax([1,1-first_dec_digit.min().astype(int)])])
            xticklabels = WOE_df.bin_min.apply(lambda x:"{:.{}f}".format(x, xtickdigits)).tolist() + ['max']   
        xlabel = "bin cutpoints"
        if orig_scale and len(WOE_df.bin_min) > 2:
            ave_bin_width = np.nanmax([(WOE_df.bin_min[2:] - WOE_df.bin_min[2:].shift()).mean(), .5])
            x_init = np.max([WOE_df.bin_min[1], WOE_df.bin_min[2] - 2*ave_bin_width])
            xticks = np.array([x_init] + WOE_df.bin_min[2:].tolist() + [WOE_df.bin_min[-1] + 2*ave_bin_width])
            xticklabels[1] = 'min'
        axs.step(xticks, [WOE_df.WOE[1]] + WOE_df.WOE[1:].tolist(), color=woeCol, label='WOE', alpha=woeAlpha)
        axs.step(xticks, [WOE_df.Score[1]] + WOE_df.Score[1:].tolist(), c='k', label='Score')

    axs.axhline(WOE_df.WOE[0], linestyle=':', color=woeCol, alpha=woeAlpha)
    axs.axhline(WOE_df.Score[0], linestyle=':', color='k')
    axs.set_title(p_var, size=18)
    axs.set_ylabel("Additive Evidence (est. marginal log-odds)")
    axs.set_xlabel(xlabel); axs.set_xticks(xticks) 
    axs.set_xticklabels(xticklabels[1:], rotation=45, ha='right', rotation_mode='anchor')
    axs.set_xlim(left=axs.get_xlim()[0]-.7*ave_bin_width)
    axs.set_ylim( ( np.where(axs.get_ylim()[0] > -1, -1, axs.get_ylim()[0])
                    , np.where(axs.get_ylim()[1] < 1, 1, axs.get_ylim()[1])) )
    if compute_stats:
        num = WOE_df.bin_ct.sum()
        n_targ = sum(WOE_df.Y1)
        MetrList = [ "# of Records = {:.0f}".format(num)
                    , "# of Targets = {:.0f}".format(n_targ)
                    , "# Missing = {:.0f}".format(WOE_df.iloc[0].bin_ct)
                    , "Base Rate = {:.3f}%".format(100*(n_targ/num))
                    , "Coefficient = {:.3f}".format(WOE_df.Coef[1])
                  ]
        ylocs = np.arange(.97,0,-.05)[:len(MetrList)]
        for yloc, sm in zip(ylocs, MetrList):
            axs.annotate(sm, xy=(1.02, yloc), xycoords='axes fraction')

    plt.legend(bbox_to_anchor=(1, 0), loc='lower left')
    plt.show()
    return(WOE_df)


# In[ ]:


def add_scorecard_points(scdf, PDO=20, standardSc_pts=600, standardSc_odds=19, pts_dec_prec=1, trndict=None):
    """Append 'Points' field to a scorecard dataframe
    
    Parameters
    ----------
    scdf : DataFrame
        A scorecard dataframe
    PDO : int, float, optional (default=20)
        Points to double odds - number of points needed for the outcome odds to double
    standardSc_pts : int, float, optional (default=600)
        Standard score points - a fixed point on the points scale with fixed odds 
    standardSc_odds : int, float, optional (default=19)
        Standard Odds - odds of good at standard score fixed point
    pts_dec_prec : int, optional (default=1)
        Decimal places to show in scorecard points 
    trndict : dictionary, optional (default=None)
        The output of woesc.describe_data_g_targ(trndat, targ_var)
        
    Returns
    -------
    pointscard : DataFrame
        The input scorecard with points column appended
    """
    factor = PDO / np.log(2)
    offset = standardSc_pts - factor * np.log(standardSc_odds)
    # print("Offset: {:.7g}, Factor: {:.6g}".format(offset, factor))

    sclSc = factor * scdf.Score  

    var_offsets = sclSc.groupby(level='var_name').max()
    ## The negative sign here flips the evidence scale (for <--> against)
    shftSc = -sclSc + var_offsets

    scdf['Points'] = shftSc.round(pts_dec_prec)
    shft_base_pts = (offset - var_offsets.sum()).round(pts_dec_prec)
    if (pts_dec_prec <= 0):
        scdf['Points'] = scdf.Points.astype(int)
        shft_base_pts = shft_base_pts.astype(int)
    
    scdf0 = pd.DataFrame([[None] * len(scdf.reset_index().columns)], columns=scdf.reset_index().columns)
    scdf0['var_name'] = "Intercept"
    scdf0['Score'] = scdf.base_score
    scdf0['Points'] = shft_base_pts
    scdf0['bins'] = 0
    scdf0['bin_idx'] = -1
    if trndict:
        scdf0['bin_ct'] = trndict['num']
        scdf0['Y1'] = trndict['n_targ']
        scdf0['targ_rate'] = trndict['base_rate']
        scdf0['WOE'] = trndict['base_log_odds']
        scdf0['Coef'] = scdf0.Score / trndict['base_log_odds'] 
        scdf0['bin_pct'] = 1        

    scdf0.set_index(scdf.index.names, inplace=True)
    pointscard = scdf0.append(scdf)
    pointscard.base_score = scdf.base_score
    pointscard.base_points = shft_base_pts
    return pointscard


# In[ ]:


## Prototype code for a univariate scorecard scikit-learn estimator

class uwoesc(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, n_cuts=2, binner='qntl', min_bin_size=100, laplace=1
                , laplaceY0='brc', compute_stats=False, neutralizeMissing=False, neutralizeSmBins=True
                , correct_cardinality=True ):
        """Univariate Weight of evidence scorecard object.
        """
        self.n_cuts = n_cuts
        self.binner = binner
        self.min_bin_size = min_bin_size
        self.laplace = laplace
        self.laplaceY0 = laplaceY0
        self.compute_stats = compute_stats  
        self.neutralizeMissing = neutralizeMissing 
        self.neutralizeSmBins = neutralizeSmBins
        self.correct_cardinality = correct_cardinality
        
        self.var_name = ''
        self.targ_var = ''
        self.base_odds = 1
        self.WOE_df = pd.DataFrame()
        self.bin_edges = []

    def fit(self, X, y, var_name):
        """ Fit function  
        """
        self.var_name = var_name
        self.targ_var = y.columns[0]
        self.base_odds = (np.sum(y==1) / np.sum(y==0))[0]
        
        dat_df = y.join(X[[self.var_name]])
        self.bin_edges, self.binner = get_bin_edges(dat_df, self.n_cuts, self.binner, self.min_bin_size
                                               , self.correct_cardinality )
        self.WOE_df = gen_uwoesc_df( dat_df, self.bin_edges, self.binner, self.n_cuts, self.min_bin_size 
                                    , self.laplace, self.laplaceY0, self.compute_stats
                                    , self.neutralizeMissing, self.neutralizeSmBins )
        return(self)        

    def transform(self, to_transform, **kwargs):  
        """ """        
        transformed = woe_score_var( to_transform, self.WOE_df, **kwargs )                  
        return(transformed)
    
    def predict(self, X):
        """ """
        score = woe_score_var( X[[self.var_name]], self.WOE_df )  
        prob = expit(np.log(self.base_odds) + score)
        return(prob)
    
    def plot(self, **kwargs):
        """Create a plot of a univariate scorecard.
        """
        return uwoesc_plot(self.WOE_df, self.targ_var, **kwargs)
    
    def custom(self, X, y, bin_edges, var_name ):
        """ Custom binning fit function        
        """
        self.var_name = var_name
        self.targ_var = y.columns[0]
        self.bin_edges = bin_edges
        self.binner = 'custom'
        self.n_cuts = 0
        self.base_odds = (np.sum(y==1) / np.sum(y==0))[0]
        
        dat_df = y.join(X[[self.var_name]])      
        self.WOE_df = gen_uwoesc_df( dat_df, self.bin_edges, self.binner, self.n_cuts, self.min_bin_size 
                                    , self.laplace, self.laplaceY0, self.compute_stats
                                    , self.neutralizeMissing, self.neutralizeSmBins )
        return(self.WOE_df)

