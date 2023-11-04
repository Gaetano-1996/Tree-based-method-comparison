
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
import warnings
from matplotlib.pyplot import subplots
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import median_absolute_error as MDAE
from sklearn.metrics import mean_squared_log_error as MSLE
from sklearn.metrics import mean_squared_error
from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,plot_tree,export_text)
from sklearn.ensemble import (RandomForestRegressor as RF,
                              GradientBoostingRegressor as GB)
from sklearn.model_selection import (GridSearchCV, 
                                     RandomizedSearchCV,
                                     KFold)


def read_data(file_name, sep=',',dir='./',verbose = True):
    """
    Function for reading a csv file.

    Parameters
    ----------
    file_name : string
        Name of the file to read.
    
    sep : string, default = ','
        Separator of the file.
    
    dir : string, default = './'
        Directory of the file.
    
    verbose : bool, default = True
        Print information about the imported dataset to the screen.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the data read from the file.
    """
    print('#'*50)
    print('#'*12,'READING FILE: ' + file_name,'#'*12)
    print('#'*50,'\n\n')

    file_name = dir + file_name
    data = pd.read_csv(file_name, sep=sep)

    if verbose:
        print('N. Observation: ', data.shape[0]) # righe del dataset
        print('N. Variables: ', data.shape[1],'\n') # colonne del dataset
        print('Info about DATASET: ')
        print('----------------------------------------')
        data.info()
        print('\nN. missing values:', data.isnull().sum().sum())
        print('----------------------------------------\n')
        print('DATASET HEAD: ')
        print('----------------------------------------')
        print(data.head(),'\n')
    
    return data   

def find_most_correlated_vars(corr, target_var, threshold=0.5):
    """
    Function for selecting the most correlated variables with the response variable.
    
    Parameters
    ----------
    corr : pandas.DataFrame
        Correlation matrix.
    
    target_var : str
        Name of the response variable.
    
    threshold : float, default = 0.5
        Correlation threshold.
    
    Returns
    -------
    list
        List containing the names of the variables most correlated with the response variable.
    """

    high_corr_vars = corr[abs(corr[target_var]) > threshold][target_var]

    sorted_vars = high_corr_vars.sort_values(key=lambda x: abs(x),ascending=False)[1:]

    return list(sorted_vars.index)

def auto_eda (data, 
              target, 
              features = 'all', 
              plot_density=True, 
              pair_plot= True, 
              cor_plot= True, 
              print_corr=True,
              verbose=True,
              threshold=0.5):
    """
    Function for exploratory data analysis.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the data.
        If the response is passed as `pd.Series`, the function processes the dataset as if it were not present in the DataFrame's columns.
    
    target : str 
        Name of the column containing the response variable.
    
    features : bool | str | list, default='all'
        If `features = all`, the function assumes that the intention is to preliminarily study the relationship between the response variable and ALL COVARIATES.
        If the relationship with one or more particular covariates is to be studied, pass the name/list of names of the corresponding columns.
    
    plot_density : bool, default=True
        Displays exploratory analysis graphs on screen.
    
    pair_plot : bool, default=True
        Displays the scatterplot matrix on screen.
    
    cor_plot : bool, default=True
        Displays the correlation matrix on screen.
    
    print_corr : bool, default=True
        Prints the correlation matrix on screen.
    
    verbose : bool, default=True
        Prints diagnostic information on screen.
    
    threshold : float, default=0.5
        Correlation threshold.
    
    Returns
    -------
    None.
    """
    warnings.filterwarnings("ignore")

    print('\n')
    print('#'*56)
    print('#'*13 + ' EDA ' + '#'*13)
    print('#'*56)
    cat = [] # categorical variables list
    cont = [] # continuous variables list
    
    if isinstance(target, str):
        data = pd.DataFrame(data)
        y = data[target]
        X = data.drop(target, axis=1)
    else: 
        print('Error: target must be a string')
        return
    
   
    if features != 'all':
        X = X[[features]]
    else:
        X = X
    

    for type,var in zip(X.dtypes,X.columns):
        if type == 'object' or type == 'category':
            cat.append(var)
        else:
            cont.append(var)

    if verbose:
            print('\n\nFEATURES SUMMARY:')
            print('-'*100)
            print(X.describe()) 

            print('-'*100)
            if y.dtypes not in ['object','category']:
                print('\TARGET SUMMARY:')
                print(y.describe())
                
            print('\nTarget variable data type:',y.dtype)
            if len(cat) > 0:
                print('\nN. Categorical variables:',cat)
                if len(cont) > 0:
                    print('\nN. Continuous variables:',cont)
            else: 
                print('\nAll variables are continuous') 
    
    if plot_density:
            sns.displot(data=data, x=target, kde = True,height=5,aspect=1) 
            plt.title('Histogram: ' + target,loc='left')
            plt.tight_layout()
            plt.show()

    if y.dtypes in ['object','category']:
        
        print('\n\nContingency Table:',target)
        print('-'*100)
        freq_table = y.value_counts() 
        print(freq_table)
        print('-'*100)
        if pair_plot:
            n_cont = len(cont)
            n_rows = int(n_cont / 2) + (n_cont % 2)
            n_cols = 2
            
            if n_cont == 0:
                print('No variables correlated with',target)
                return
            elif n_cont == 1:
                n_rows = 1
                n_cols = 1
            elif n_cont == 3:
                n_rows = 3
                n_cols = 1
            elif n_cont == 5:
                n_rows = 5
                n_cols = 1
            elif n_cont == 7:
                n_rows = 7
                n_cols = 1
            elif n_cont == 9:
                n_rows = 3
                n_cols = 3
            elif n_cont == 16:
                n_rows = 4
                n_cols = 4
            
            axs = subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows),layout='constrained')[1]
            for i, var in enumerate(cont):
                row = int(i / n_cols)
                col = i % n_cols
                sns.boxplot(x=var, y=target, data=data, ax=axs[row, col])
                axs[row, col].set_xlabel(var)
                axs[row, col].set_ylabel(target)
                axs[row, col].set_title('Boxplot: ' + target)
            plt.show()
        if verbose: 
            for var in cat:
                print('\nDOUBLE FREQUENCY TABLE:',target,'Vs.',var)
                print('-'*100)
                print(pd.crosstab(y,X[var]),'\n')
    else: 
        corr_mtx = data[[target] + cont].corr()
        if print_corr: 
            print('\nCORRELATION MATRIX:')
            print('-'*100)
            print(corr_mtx,'\n')
        if cor_plot: 
            ax = subplots(figsize=(8,8))[1]
            sns.heatmap(corr_mtx, 
                        annot=True, 
                        cmap='Blues', 
                        cbar=True,fmt='.2f',mask=False,ax=ax)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.show()
        
        if pair_plot:
            vars_corr = find_most_correlated_vars(corr_mtx, target_var=target, threshold=threshold)
            n_var = len(vars_corr)
            n_rows = int(n_var / 2) + (n_var % 2)
            n_cols = 2

            
            if n_var == 0:
                print('No variables correlated with',target)
                return
            elif n_var == 1:
                n_rows = 1
                n_cols = 1
            elif n_var == 3:
                n_rows = 1
                n_cols = 3
            elif n_var == 5:
                n_rows = 1
                n_cols = 5
            elif n_var == 7:
                n_rows = 1
                n_cols = 7
            elif n_var == 9:
                n_rows = 3
                n_cols = 3
            elif n_var == 16:
                n_rows = 4
                n_cols = 4

            axs = subplots(n_rows, n_cols, squeeze=False,figsize=(7*n_cols, 5*n_rows),layout='constrained')[1]
            for i, var in enumerate(vars_corr):
                row = int(i / n_cols)
                col = i % n_cols
                sns.regplot(data=data,x=var, y=target,lowess=True,ax=axs[row, col],
                            scatter_kws={'alpha':0.5},
                            line_kws={'color':'C1','lw':2.5})
                axs[row, col].set_xlabel(var)
                axs[row, col].set_ylabel(target)
                axs[row, col].set_title('Corr: ' + str(round(corr_mtx.loc[var,target],2)))
            plt.show()
   
def model_matrix(data,
                 remove=False,
                 target=False,
                 intercept=False,
                 dummies=False,
                 verbose=True):
    """
    Function for creating the model matrix (or design matrix).
    Specifically, the function returns the matrix of covariates,
    by optionally adding the intercept and converting categorical variables 
    into dummies if requested.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        DataFrame containing the data (excluding the response variable if `target = False`).
    
    remove : str | list, default=False
        Name or list of names of variables to NOT include in the model matrix.
    
    target : bool | str, default=False
        If `target = False`, the function assumes that the data is passed without the response variable.
        If data is passed with the response variable, pass the name or index of the column to `target`.
    
    intercept : bool, default=False
        Adds the intercept to the model matrix.
    
    dummies : bool, default=False
        Converts categorical variables into dummies.
        The default behavior is consistent with the R language, we create c dummies where c is the number of 
        categories and use c-1 of them for the model.
    
    verbose : bool, default=True
        Prints diagnostic information on screen.
    
    Returns
    -------
    pandas.DataFrame
        The model matrix.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data) 
    
    if target:
        if isinstance(target, str): 
            X = data.drop(target, axis=1,inplace=False)
        else: 
            X = data.drop(data.columns[target], axis=1,inplace=False)
    else: 
        X = data 
    
    if intercept:
        X = pd.concat([pd.Series(1, index=X.index, name='Intercept'), X], axis=1)
    
    if dummies:
        # selezioniamo le variabili categoriali
        for col in X.select_dtypes(include=['object']).columns:
            # calcoliamo il valore più frequente della variabile
            base = X[col].value_counts().index[0]
            # rimuoviamo il valore più frequente che terremo come base
            X[col] = X[col].replace(base, np.nan)
            # codifichiamo i valori in dummies
            dummies = pd.get_dummies(X[col], prefix=col)
            # aggiungiamo le nuove colonne alla matrice
            X = pd.concat([X, dummies], axis=1)
            # rimuoviamo la colonna originale
            X = X.drop(col, axis=1)
            # codifichiamo True = 1 e False = 0
            data = data.replace(True,1)
            data = data.replace(False,0)

    # rimozione delle variabili
    if remove:
        X.drop(remove, axis=1, inplace=True)
        
    # Diagnostica
    if verbose:
        print('\nCovariates in the model matrix: ')
        print('----------------------------------------')
        print(list(X.columns),'\n')
    return X

def split_data (data,target,size=.25,seed=0,verbose=True):
    """
    Function for splitting the dataset into a train and test set.

    Parameters
    ----------
    data : pandas.DataFrame or ND-array
        DataFrame containing the data or the model matrix generated with the `model_matrix` function.
    
    target : str | pandas.Series
        Name or index of the column containing the response variable or the series containing the response variable.
    
    size : float, default=.25
        Percentage of observations to be assigned to the test set.
    
    seed : int, default=0
        Seed for the random number generation.
    
    verbose : bool, default=True
        Prints diagnostic information to the screen.
        
    Returns
    -------
    pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series
        Train and test set model matrices and train and test set response series.
    """

    if isinstance(target, str): 
        y = data[target]
        X = data.drop(target, axis=1)
    else: 
        y = target
        X = data
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=size, 
                                                        random_state=seed)
    
    # Diagnostica
    if verbose:
        print('\nSplitting data:')
        print('----------------------------------------')
        print('> N. observation (Train-set): ', len(X_train))
        print('> N. observation (Test-set): ', len(X_test))
    
    return X_train, X_test, y_train, y_test

def MSE(y, y_hat):
    """
    Function for calculating the mean squared error.

    Parameters
    ----------
    y : array
        Array containing the observed values.
    
    y_hat : array
        Array containing the predicted values.

    Returns
    -------
    float
        Mean squared error.
    """
    return np.mean((y - y_hat)**2)

def MAE(y, y_hat):
    """
    Function for calculating the mean absolute error.

    Parameters
    ----------
    y : array
        Array containing the observed values.
    
    y_hat : array
        Array containing the predicted values.

    Returns
    -------
    float
        Mean absolute error.
    """
    return np.mean(np.abs(y - y_hat))

def perf(y_test,
        y_hat,
        metodo=None,
        scoring='all',
        plot=False):
    """
    Function for calculating the performance metrics of the model.

    Parameters
    ----------
    y_test : array
        Array containing the observed values.
    
    y_hat : array
        Array containing the predicted values.
    
    metodo : str, default=None
        Name of the method used to generate predictions.
    
    scoring : str, default='all'
        Performance metric to calculate.
        Possible metrics are: `'mse'`, `'rmse'`, `'mae'`, `'mape'`, `'mdae'`, `'msle'`.
        If `scoring = 'all'`, all metrics are calculated.
    
    plot : bool, default=False
        Prints the scatter plot between observed and predicted values to the screen.
    
    Returns
    -------
    None.
    """
    # selezioniamo la metrica di performance 
    if scoring == 'mse':
        print(f'Performance metric: MSE =', MSE(y_test, y_hat))
    elif scoring == 'rmse':
        print(f'Performance metric: RMSE = ', np.sqrt(MSE(y_test, y_hat)))
    elif scoring == 'mae':
        print(f'Performance metric: MAE = ', MAE(y_test, y_hat))
    elif scoring == 'mape':
        print(f'Performance metric: MAPE = ', MAPE(y_test, y_hat))
    elif scoring == 'mdae':
        print(f'Performance metric: MDAE = ', MDAE(y_test, y_hat))
    elif scoring == 'msle':
        print(f'Performance metric: MSLE = ', MSLE(y_test, y_hat))
    else:
        print('\nREGRESSION PERFORMANCE:')
        print('----------------------------------------')
        print('> MSE: ', round(MSE(y_test, y_hat),3))
        print('> RMSE:', round(np.sqrt(MSE(y_test, y_hat)),3))
        print('> MAE: ', round(MAE(y_test, y_hat),3))
        print('> MAPE:', round(MAPE(y_test, y_hat),3))
    # Grafico dei valori previsti contro i valori osservati, 
    # più le previsioni sono accurate, più i punti saranno allineati lungo la bisettrice
    if plot:
        sns.regplot(x=y_test, 
                    y=y_hat,
                    scatter_kws={'color':'black','alpha':0.3},
                    line_kws={'color':'red','lw':2,'alpha':0.9})
        
        # reale bisettrice
        x = np.linspace(np.min(y_test), np.max(y_test), 100)
        y = x
        plt.plot(x, y, 'b--', label='Bisector')
        # se pasato aggiungiamo al titolo il nome del metodo che ha generato le previsioni
        if metodo:
            plt.title(f'{metodo.capitalize()}: y_test Vs. y_hat',loc='left')
        else:
            plt.title(f'y_test Vs. y_hat')
        plt.xlabel('y_test')
        plt.ylabel('y_hat')
        plt.show()

def method_comparison(X_train, X_test, y_train, y_test,
                      models,
                      train = False,
                      max_estimators=200
                      ):
    """
    Function for comparing ensemble regression models.
    The function is particularly computationally demanding, therefore a number 
    of max_estimators that is not too high is recommended.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Train set model matrix.
    
    X_test : pandas.DataFrame
        Test set model matrix.
    
    y_train : pandas.Series
        Train set response series.
    
    y_test : pandas.Series
        Test set response series.
    
    models : dict
        Dictionary containing the ensemble models to compare.
        The dictionary key is the model name and the value is the model object.
    
    train : bool, default=False
        If `train = True`, the comparison is performed on both the test set and the train set.
        If `train = False`, the comparison is performed only on the test set, which can reduce computational complexity.
    
    max_estimators : int, default=200
        The maximum number of estimators to compare.

    Returns
    -------
    None.
    """
    
    # liste per errori di train e test
    bagging_train_error, bagging_test_error = [], []
    rf_train_error, rf_test_error = [], []
    boosting_train_error, boosting_test_error = [], []
    # liste per tempi di adattamento
    bagging_fit_time, rf_fit_time, boosting_fit_time, mean_fit_time = [], [], [], []
    # Timer per il calcolo del tempo di esecuzione dell'intero confronto tra modelli
    start = time()
    # ignoriamo i warning derivanti dai grafici 
    warnings.filterwarnings("ignore")
    for i in range(1, max_estimators+1): # iteriamo lungo il numero di stimatori
        for name, model in models.items(): # iteriamo lungo i modelli (dizionario nome, oggetto)
            model.set_ensemble_params(n_estimators=i) # impostiamo il numero di stimatori del modello in questione
            fit_time = model.auto_fit(X_train, y_train,verbose =False)[0]
            # Per ogni modello andiamo a salvare il tempo richiesto dall' adattamento
            if name == 'bagging':
                bagging_fit_time.append(fit_time)
            elif name == 'rf':
                rf_fit_time.append(fit_time)
            elif name == 'boosting':
                boosting_fit_time.append(fit_time)
            # calcoliamo l'errore sul training set
            if train:
                y_train_pred = model.predict(X_train)
                if name == 'bagging':
                    bagging_train_error.append(mean_squared_error(y_train, y_train_pred))
                elif name == 'rf':
                    rf_train_error.append(mean_squared_error(y_train, y_train_pred))
                elif name == 'boosting':
                    boosting_train_error.append(mean_squared_error(y_train, y_train_pred))
            # calcoliamo l'errore sul test set
            y_test_pred = model.predict(X_test)
            if name == 'bagging':
                bagging_test_error.append(mean_squared_error(y_test, y_test_pred))
            elif name == 'rf':
                rf_test_error.append(mean_squared_error(y_test, y_test_pred))
            elif name == 'boosting':
                boosting_test_error.append(mean_squared_error(y_test, y_test_pred))
    # Andamento dell'errore su training/test set
    plt.figure(figsize=(8, 8))
    if train:
        plt.plot(range(1, max_estimators+1), bagging_train_error, 'b', label='Bagging Train Error')
    plt.plot(range(1, max_estimators+1), bagging_test_error, 'r', label='Bagging Test Error')
    if train:
        plt.plot(range(1, max_estimators+1), rf_train_error, 'g', label='Random Forest Train Error')
    plt.plot(range(1, max_estimators+1), rf_test_error, 'm', label='Random Forest Test Error')
    if train:
        plt.plot(range(1, max_estimators+1), boosting_train_error, 'c', label='Boosting Train Error')
    plt.plot(range(1, max_estimators+1), boosting_test_error, 'y', label='Boosting Test Error')
    plt.xlabel('Number of Estimators')
    plt.ylabel('MSE')
    plt.ylim(-1,40)
    if train:
        plt.title('Training and Test Error vs Number of Estimators')
    else:
        plt.title('Test Error vs Number of Estimators')
    plt.legend(loc='upper right')
    plt.show()

    # Andamento del tempo di adattamento
    plt.figure(figsize=(8, 8))
    plt.plot(range(1, max_estimators+1), bagging_fit_time, 'b', label='Bagging Fitting Time')
    plt.plot(range(1, max_estimators+1), rf_fit_time, 'g', label='Random Forest Fitting Time')
    plt.plot(range(1, max_estimators+1), boosting_fit_time, 'c', label='Boosting Fitting Time')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Fitting Time (seconds)')
    if train:
        plt.title('Fitting Time vs Number of Estimators')
    else:
        plt.title('Test Fitting Time vs Number of Estimators')
    plt.legend(loc='upper left')
    plt.show()
    # diagnostica sul tempo impiegato dal confronto
    print("Time: ", round((time() - start)/60), " ~minutes")
