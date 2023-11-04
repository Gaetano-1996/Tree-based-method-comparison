from src.AutoTree import AutoTree
from src.AutoEnsemble import AutoEnsemble
from src.utils import *
# Analisi dati Boston Housing ################################################################################################################################################

def boston_analysis():
    # lettura dei dati da csv
    dati = read_data("Boston.csv")
    # analisi esplorativa dei dati
    auto_eda(dati,
            'medv',
            print_corr=False)
    # Selezione della variabilerisposta
    y = dati['medv']
    # Creazione matrice del modello
    X = model_matrix(dati,'medv')
    # definizione di un oggetto della classe AutoTree come albero di regressione
    reg = AutoTree('regressione')
    # diagnostica sull'albero inizializzato
    print(reg)  
    # Divisione del dataset in insieme di stima ed insieme di verifica
    X_train, X_test, y_train, y_test = split_data(X, y,size = 0.25)
    # Primo adattamento del modello ai dati
    reg.auto_fit(X_train,y_train,verbose = 2,plot=True)
    # Valori stimati dal modello
    previsioni = reg.predict(X_test)
    # Valutazione delle performance del modello
    perf(y_test, previsioni)
    # Potatura automatica dell'albero di regressione
    auto_pruned = reg.auto_prune(X_train, y_train,auto=True,plot=False)
    print(auto_pruned)
    # Adattamento dell'albero con valore di alpha selezionato in modo automatico
    auto_pruned.auto_fit(X_train, y_train, verbose=1, plot=False)
    # valori previsti
    previsioni2 = auto_pruned.predict(X_test)
    # valutazione della performance del modello potato automaticamente
    perf(y_test, previsioni2)
    # Potatura "manuale"
    manually_pruned = reg.auto_prune(X_train, y_train,auto=False)
    # Adattamento dell'albero potato manualmente
    manually_pruned.auto_fit(X_train, y_train, verbose=1, plot=True)
    # Previsione e valutazione delle performance
    previsioni3 = manually_pruned.predict(X_test)
    perf(y_test,previsioni3,plot=True,metodo="Albero di regressione (potato)")
    # inizializziamo l'ensemble con bagging
    bag = AutoEnsemble(method = 'rf', n_estimators = 100, random_state = 0) # usiamo 'bagging' ma in realta il default e' 'rf'
    # fit dei dati
    bag.auto_fit(X_train, y_train)
    # predizione + performance
    bag_pred = bag.predict(X_test)
    perf(y_test, bag_pred)
    # tuning dei parametri : B non è un parametro fondamentale, basta scegliere un valore che non sia troppo piccolo
    parametri_bag = {
            'n_estimators': [100, 500, 1000],
            'max_depth': [3, 5,10,15,20],
            'min_samples_split': [1,2, 5, 10],
            'min_samples_leaf': [1, 2, 5]
        }

    bag2 = bag.auto_tuning(
        X_train,
        y_train,
        parametri_bag,
        n_splits=5,
        n_iter=10,
        randomized=True
    )

    # performance modello ottimizzato
    bag_pred2 = bag2.predict(X_test)
    perf(y_test, bag_pred2,metodo='Bagging',plot=True)
    # feature selection del modello bagging
    bag2.auto_features(X_train) 
    # inizializzazione con RF con i predittori selezionati 
    rf = AutoEnsemble(max_features=6)
    # adattamento ai dati di training
    rf.auto_fit(X_train, y_train)
    # predict + performance
    rf_pred = rf.predict(X_test)
    perf(y_test, rf_pred)
    # tuning 
    parametri_rf = {'n_estimators': [200,500,1000],
                    'criterion':['squared_error','friedman_mse'],
                    'max_features': [None,6,'sqrt','log2']}
    rf2 = rf.auto_tuning(X_train,y_train,parametri_rf,n_splits=5,n_iter=10,random_state=0)
    # Performance del modello ottimizzato
    rf_pred2 = rf2.predict(X_test)
    perf(y_test, rf_pred2,metodo='Random Forest',plot=True)
    # Importanza dei predittori (Random Forest)
    rf2.auto_features(X_train)
    # Inizializzazione
    boost = AutoEnsemble('gb')
    # adattamento
    boost.auto_fit(X_train, y_train)
    # previsione
    y_pred = boost.predict(X_test)
    # performance
    perf(y_test, y_pred)
    # tuning
    # tuning
    parametri_boost ={
        'max_depth': [1,2,3],
        'n_estimators': [500, 1000, 5000],
        'learning_rate': [0.001, 0.01, 0.05, 0.1,0.2],
        'criterion': ['friedman_mse', 'squared_error']
    }

    boost2 = boost.auto_tuning(X_train, y_train, parametri_boost,n_splits=5, n_iter=10,random_state=0)
    pred2 = boost2.predict(X_test)
    perf(y_test,pred2,plot=True,metodo='Boosting')
    # confronto tra i metodi 
    method_comparison(X_train,
                    X_test,
                    y_train,
                    y_test,{'bagging':bag2,"rf":rf2,"boosting":boost2},train=True,max_estimators=500)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    boston_analysis()