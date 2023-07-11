
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()




from sklearn.model_selection import train_test_split

pd.options.display.max_columns = None


############################## credit risk modeling
loan_data_backup = pd.read_csv('loan_data_2007_2014.csv')
loan_data = loan_data_backup.copy()
# pd.options.display.max_columns = None

# preprocessing continuous variables
loan_data.head()
loan_data.tail()
loan_data.columns.values
loan_data.info()
loan_data['emp_length'].unique()
loan_data['emp_length_int'] = loan_data['emp_length'].str.replace('\+ years', '')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('< 1 year', str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].fillna(str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' years', '')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' year', '')
loan_data['emp_length_int'] = pd.to_numeric(loan_data['emp_length_int'])
type(loan_data['emp_length_int'][0])

loan_data['earliest_cr_line_date'] =  pd.to_datetime(loan_data['earliest_cr_line'], format = "%b-%y")
type(loan_data['earliest_cr_line_date'][0])
pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date'] 
loan_data['mths_since_earliest_cr_line'] = round((pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date']) / np.timedelta64(1, 'M'))
loan_data['mths_since_earliest_cr_line'].describe()
loan_data.loc[:, ['earliest_cr_line','earliest_cr_line_date', 'mths_since_earliest_cr_line' ]][loan_data['mths_since_earliest_cr_line']<0]
loan_data['mths_since_earliest_cr_line'][loan_data['mths_since_earliest_cr_line']<0] = loan_data['mths_since_earliest_cr_line'].max()
loan_data['mths_since_earliest_cr_line'].describe() # after imputing, there is no negative anymore

loan_data['term'] = loan_data['term'].str.replace('months', '')
loan_data['term_int'] = loan_data['term'].astype(int)

loan_data['issue_d_date'] = pd.to_datetime(loan_data['issue_d'], format = "%b-%y")
loan_data['mths_since_issue_d'] = round((pd.to_datetime('2017-12-01') - loan_data['issue_d_date']) / np.timedelta64(1, 'M'))
loan_data['mths_since_issue_d'].describe()

# preprocessing discrete variables
loan_data_dummies = [pd.get_dummies(loan_data['grade'], prefix = 'grade'),
                     pd.get_dummies(loan_data['sub_grade'], prefix = 'sub_grade'),
                     pd.get_dummies(loan_data['home_ownership'], prefix = 'home_ownership'),
                     pd.get_dummies(loan_data['verification_status'], prefix = 'verification_status'),
                     pd.get_dummies(loan_data['loan_status'], prefix = 'loan_status'),
                     pd.get_dummies(loan_data['purpose'], prefix = 'purpose'),
                     pd.get_dummies(loan_data['addr_state'], prefix = 'addr_state'),]



# missing values and cleaning
pd.options.display.max_rows = None
loan_data.isnull().sum()

pd.options.display.max_rows = 100
loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'], inplace = True )
# loan_data['total_rev_hi_lim'].isnull().sum()

loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean(), inplace  = True)

loan_data['mths_since_earliest_cr_line'].fillna(0, inplace=True)
loan_data['acc_now_delinq'].fillna(0, inplace=True)
loan_data['total_acc'].fillna(0, inplace=True)
loan_data['pub_rec'].fillna(0, inplace=True)
loan_data['open_acc'].fillna(0, inplace=True)
loan_data['inq_last_6mths'].fillna(0, inplace=True)
loan_data['delinq_2yrs'].fillna(0, inplace=True)
loan_data['emp_length_int'].fillna(0, inplace=True)

# calculating expected loss (EL) = PD * LGD * EAD
# probability of defualt (PD) * loss given default (LGD) (proportion) * Exposure at default (ED) (amount) 

# DV definition: 0 is default, 1 is not defualt
loan_data['loan_status'].unique()
loan_data['loan_status'].value_counts()
loan_data['loan_status'].value_counts() / loan_data['loan_status'].count()
loan_data['good_bad'] = np.where(loan_data['loan_status'].isin(['Charged Off', 'Late (31-120 days)', 'Default', 'Does not meet the credit policy. Status:Charged Off']), 0 ,1)
loan_data['good_bad']

## data preparation
# splitting dataset for train and test

loan_data.columns

X_train, X_test, y_train, y_test = train_test_split(loan_data.drop('good_bad', axis = 1), loan_data['good_bad'], test_size = 0.2, random_state= 42)

df_inputs_prepr = X_train
df_targets_prepr = y_train

# create a new dataframe with only on IV (grade) and DV (good or bad borrower)
df_inputs_prepr['grade'].unique()
df1 = pd.concat([df_inputs_prepr['grade'], df_targets_prepr], axis = 1)
df1.head()

df1.groupby(df1.columns.values[0], as_index = False)[df1.columns.values[1]].count()

# calculate proportion of good borrower
df1.groupby(df1.columns.values[0], as_index = False)[df1.columns.values[1]].mean()

# merge these two results into one dataframe
df1= pd.concat([df1.groupby(df1.columns.values[0], as_index = False)[df1.columns.values[1]].count(), \
                df1.groupby(df1.columns.values[0], as_index = False)[df1.columns.values[1]].mean()],axis = 1)
df1 = df1.iloc[:,[0,1,3]]
df1.columns = [df1.columns.values[0], 'n_obs', 'prop_good']
# create a new col for proportion of obs in each grade
df1['prop_n_obs'] = df1['n_obs'] / df1['n_obs'].sum()
# calculate the number of good borrower and bad borrower
df1['n_good'] = df1['prop_good'] * df1['n_obs']
df1['n_bad'] = (1- df1['prop_good']) * df1['n_obs']

df1['prop_n_good'] = df1['n_good'] / df1['n_good'].sum()
df1['prop_n_bad'] = df1['n_bad'] / df1['n_bad'].sum()

# calculate weight of evidence  (WoE)
df1['WoE'] = np.log(df1['prop_n_good'] / df1['prop_n_bad'] )

# sort the rows let the highest rate of default on the top
df1 = df1.sort_values(['WoE'])
df1 = df1.reset_index(drop=True)
df1['diff_prop_good'] = df1['prop_good'].diff().abs()
df1['diff_WoE'] = df1['WoE'].diff().abs()

# calculate information value
pd.options.display.max_columns = None
df1['IV'] = (df1['prop_n_good'] - df1['prop_n_bad'] ) * df1['WoE']

df1['IV'] = df1['IV'].sum()
df1
##############################################################
# data preparation for discrete variable with function
def woe_discrete(df, discrete_variable_name, goog_bad_variable_df):
    df = pd.concat([df[discrete_variable_name], goog_bad_variable_df], axis = 1)
    df= pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(), \
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:,[0,1,3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1- df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'] )
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop=True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad'] ) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df

df_temp = woe_discrete(df_inputs_prepr, 'grade', df_targets_prepr)
df_temp
df1.equals(df_temp) # check the funciton is correct, so the ouput of df1 and df_temp will be the same

# visulalizing results with funcitons
def plot_by_woe(df_woe, rotation_of_x_labels = 0):
    x = np.array(df_woe.iloc[:,0].aaply(str))
    y = df_woe['WoE']
    plt.figure(figsize=(18,6))
    plt.plot(x,y, marker ='o', linestyle = '--', color = 'k')
    plt.xlabel(df_woe[0])
    plt.ylabel('Weight of Evidence')
    plt.title(str('Weight of Evidence' + df_woe.columns[0]))
    plt.xticks(rotation = rotation_of_x_labels)
    


