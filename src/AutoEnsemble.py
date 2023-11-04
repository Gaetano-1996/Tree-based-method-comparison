from src.utils import *
# Classe AutoEnsemble ########################################################################################################################################################
class AutoEnsemble:
   """
    Class that implements an ensemble of regression models.
    Methods can be of three types: bagging, random forest, and boosting.
    Methods are implemented using the sklearn library.

    Parameters
    ----------
    method : str, default='rf'
        The ensemble method to use. Can be 'rf' for random forest or 'gb' for gradient boosting.

    n_estimators : int, default=100
        If `method = 'rf'`: the number of trees to grow in parallel.
        If `method = 'gb'`: the number of boosting stages to perform
        (simply put: the number of trees to grow in sequence).

    criterion : str, default='squared_error'
        The criterion to measure the quality of a split, used for feature selection.
        For bagging and random forest models it can be 'squared_error', 'friedman_mse', 'absolute_error', or 'poisson'.
        For the boosting model it can be 'squared_error', 'friedman_mse'
        (Friedman's improvement is generally preferable, the default behavior makes the methods comparable).

    max_depth : int, default=3
        The maximum depth of the individual regression trees.
        If None, nodes are expanded until all leaves are pure or 
        contain fewer than `min_samples_split` samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split at any depth will only be considered if it leaves at least 
        `min_samples_leaf` training samples in each of the left and right branches.

    max_features : {'sqrt', 'log2', None}, int, float, default=None
        The number of features to consider when looking for the best split:
        - If `int`, then consider `max_features` features at each split;
        - If `float`, then `max_features` is a fraction and `max(1, int(max_features * n_features))` features are considered at each split;
        - If 'sqrt', `max_features=sqrt(n_features)`;
        - If 'log2', `max_features=log2(n_features)`;
        - If None, `max_features=n_features`.

        .. note::
        If method = 'rf', the default `max_features=None` or 1.0 is equivalent to Bagging.
        If method = 'gb', choosing `max_feature < n_features` reduces variance at the expense of bias.
        However, the search for the best split does not stop until at least one valid split is found,
        even if it requires checking more than `max_features` features.
    
    max_leaf_nodes : int, default=None
        The algorithm grows trees with `max_leaf_nodes` number of leaves or fewer.
        If None, there are no limits on the number of leaves.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
        
    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble,
        otherwise, just fit a whole new ensemble.
    
    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning.
        The subtree with the largest complexity that's less than `ccp_alpha` will be chosen.

    random_state : int, RandomState instance or None, default=0
        If method = 'rf': controls the randomness of the bootstrapping and the feature sampling for the best split at each node.
        If method = 'gb': controls the random seed given to each tree at each boosting iteration.
    
    n_jobs : int, default=-1
        The number of jobs to run in parallel for both `fit` and `predict`.
        Only used if `method` is set to `rf`.
    
    max_samples : int or float, default=None
        If None, `max_samples` is set to the number of samples in X.
        If `int`, `max_samples` is the number of samples to draw from X to train each base estimator.
        If `float`, `max_samples` is a fraction and `max(1, round(max_samples * n_samples))` samples are used.
        Only used if `method` is set to `rf`.
    
    learning_rate : float, default=0.001
        Rate at which the contribution of each tree is shrunk, `learning_rate` factor.
        Only used if `method` is set to `gb`.
    
    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base learners.
        Only used if `method` is set to `gb`.
    
    model : sklearn.ensemble, default=None
        The ensemble model to use.
        If None, the model specified by `method` and other parameters is used.
    
    Disclaimer
    ---------- 
    This implementation is only intended for regression problems.
"""

    def __init__(
            self,
            method = 'rf',
            n_estimators=100,
            criterion = "squared_error",
            max_depth = None,
            min_samples_split = 2,
            min_samples_leaf = 1,
            max_features = None,
            max_leaf_nodes = None,
            min_impurity_decrease = 0.0,
            warm_start = False,
            ccp_alpha = 0.0,
            random_state = 0,
            n_jobs = -1, # solo bagging e rf
            max_samples = None, # solo baggin e rf
            learning_rate = 0.001, # solo boosting
            subsample = 1.0, # solo boosting
            model = None
        ):
        self.method = method
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.warm_start = warm_start
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.max_samples = max_samples
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.model = model

        # Gradient Boosting
        if self.method == 'gb':
            # i comportamenti per i parametri non menzionati (loss, init, alpha, verbose,
            # validation_fraction, n_iter_no_change e tol) sono quelli di default
            self.model = GB(
                learning_rate = self.learning_rate,
                n_estimators = self.n_estimators,
                subsample = self.subsample,
                criterion = self.criterion,
                max_depth = self.max_depth,
                min_samples_split = self.min_samples_split,
                min_samples_leaf = self.min_samples_leaf,
                max_features = self.max_features,
                max_leaf_nodes = self.max_leaf_nodes,
                min_impurity_decrease = self.min_impurity_decrease,
                warm_start = self.warm_start,
                ccp_alpha = self.ccp_alpha,
                random_state = self.random_state
            )
        # Bagging o Random Forest
        else:
            # i comportamenti per i parametri non menzionati (bootstrap, oob_score e monotonic_cst)
            # sono quelli di default
            self.model = RF(
                n_estimators = self.n_estimators,
                criterion = self.criterion,
                max_depth = self.max_depth,
                min_samples_split = self.min_samples_split,
                min_samples_leaf = self.min_samples_leaf,
                max_features = self.max_features,
                max_leaf_nodes = self.max_leaf_nodes,
                min_impurity_decrease = self.min_impurity_decrease,
                warm_start = self.warm_start,
                ccp_alpha = self.ccp_alpha,
                random_state = self.random_state,
                n_jobs = self.n_jobs,
                max_samples = self.max_samples
            )
        if self.method != 'gb' and self.max_features is None:
            print(f"\n{'#'*5} Ensemble method: Bootstrap aggregating! Successfully initialized! {'#'*5}")
        elif self.method != 'gb' and self.max_features is not None:
            print(f"\n{'#'*5} Ensemble method: Random Forest! Successfully initialized! {'#'*5}")
        else:
            print(f"\n{'#'*5} Ensemble method: Boosting! Successfully initialized! {'#'*5}")

    
    def __repr__(self):
        """
        Method that establishes the string representation of the object.
        """
        if self.method == 'rf' and self.max_features is None:
            return f"Method: BAGGING, \n \
            Number of bootstrap aggregated samples : {self.n_estimators}, \n \
            Split criterion : {self.criterion}', \n \
            Maximum depth of single tree : {self.max_depth}, \n \
            min_samples_split={self.min_samples_split}, \n \
            min_samples_leaf={self.min_samples_leaf}, \n \
            Maximum number of features extracted for each split : {self.max_features}."
        elif self.method == 'rf' and self.max_features is not None:
            return f"Method: RANDOM FOREST, \n \
            Number of bootstrap aggregated samples : {self.n_estimators}, \n \
            Split criterion : {self.criterion}', \n \
            Maximum depth of single tree : {self.max_depth}, \n \
            min_samples_split={self.min_samples_split}, \n \
            min_samples_leaf={self.min_samples_leaf}, \n \
            Maximum number of features extracted for each split : {self.max_features}."
        else:
            return f"Method: GRADIENT BOOSTING, \n \
            Number of iterations : {self.n_estimators}, \n \
            Split criterion : {self.criterion}', \n \
            Regularization parameter : {self.learning_rate}, \n \
            Maximum depth of single tree : {self.max_depth}, \n \
            min_samples_split={self.min_samples_split}, \n \
            min_samples_leaf={self.min_samples_leaf}, \n \
            Maximum number of features extracted for each split : {self.max_features}."

    def set_ensemble(self, model):
        """
        Metodo per impostare un modello di ensemble diverso da quello di default.
        
        Parameters
        ----------
        model : sklearn.ensemble
            Modello di ensemble da utilizzare.
        
        Returns
        -------
        self : object
        """
        self.model = model
        return self
    
    def set_ensemble_params(self, **params_dict):
        """
        Metodo per impostare i parametri del modello di ensemble.

        Parameters
        ----------
        **params_dict : dict
            Dizionario che associa ad ogni iper-parametro un valore.
        
        Returns
        -------
        self : object
        """
        self.model.set_params(**params_dict)
        return self

    def fit(self,X,y):
        """
        Metodo per addestrare il modello di ensemble.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vettore delle features.
        
        y : array-like of shape (n_samples, n_targets)
            Vettore delle labels.
        
        Returns
        -------
        self : object
        """
        self.model.fit(X,y)
    
    def predict(self,X):
        """
        Metodo per effettuare predizioni sulle features passate.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vettore delle features.
        
        Returns
        -------
        y_pred : array-like of shape (n_samples, n_targets)
            Vettore delle predizioni.
        """
        return self.model.predict(X)
    
    def auto_fit(self,X,y,verbose=True):
        """
        Metodo per addestrare il modello di ensemble, restituisce il tempo impiegato per l'addestramento.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vettore delle features.
        
        y : array-like of shape (n_samples, n_targets)
            Vettore delle labels.

        verbose : bool, default=True
            Se True stampa il tempo impiegato per l'addestramento.
        
        Returns
        -------
        self : object
        """
        start = time()
        self.fit(X,y)
        if verbose:
            print(f"\n Tempo impiegato per l'addestramento: {round(time()-start,3)} secondi.")
        return (round(time()-start,3)),self

    def auto_tuning(self,
                    X_train,
                    y_train,
                    params_dict,
                    scoring='neg_mean_squared_error',
                    n_splits=5,
                    randomized=True,
                    force = False,
                    n_iter=10,
                    random_state=0,
                    verbose=0,
                    n_jobs=-1):
       """
        Function that implements the search for the optimal hyperparameters for the given model
        via cross-validation over a list of values associated with them.

        The search can be performed through:

        + grid search: exhaustive search over all provided values as the hyperparameter space in full brute-force style
        (we evaluate through cross-validation all possible combinations of hyperparameters and keep the best);

        + random search: search on a subset randomly drawn from the values provided as the hyperparameter space.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training features vector.

        y_train : array-like of shape (n_samples, n_targets)
            Training labels vector.

        params_dict : dict
            Dictionary that associates each hyperparameter with a list of values to test.

        scoring : str, callable, list/tuple, dict or None, default='neg_mean_squared_error'
            Scoring function to use for selecting the best model.

        n_splits : int, default=5
            Number of folds to use in cross-validation.
            
        randomized : bool, default=False
            If True, the search for optimal hyperparameters is performed via random search,
            otherwise, it is performed via grid search.
            
        force : bool, default=False
            If True, forces the search for optimal hyperparameters despite the number of combinations to try being greater than 500.
            Otherwise, the search is interrupted, and an error message is printed.

        n_iter : int, default=10
            Number of iterations to perform in the random search for each train-test partition of the CV.
            Expresses the number of random combinations of hyperparameters to test.
            Ignored if `randomized = False`.

        random_state : int, RandomState instance or None, default=0
            The random seed used for generating the k partitions and random search.
            
        verbose : int, default=0
            If `1`, prints the number of combinations to test to the screen.
            If `>1`, prints the number of combinations to test and diagnostics for each search iteration to the screen.
            
        n_jobs : int, default=-1
            Number of cores to use in parallel during the search for optimal hyperparameters.

        Returns
        -------
        self : object
        """

        print('\n')
        print('#'*56)
        print('#'*16 + ' HYPER-PARAMETER TUNING ' + '#'*16)
        print('#'*56)
        # k-fold
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        # setting del processo di ricerca
        # tipo di ricerca
        if randomized: # ricerca stocastica
            iterazioni = n_splits * min(n_iter,np.prod([len(params_dict[key]) for key in params_dict.keys()]))
            if iterazioni > 1000 and not force: # elevato numero di combinazioni senza forzare la ricerca
                raise ValueError("Troppe iterazioni (%d)! Prova a ridurre il numero di combinazioni di parametri o di partizioni da convalidare."%iterazioni)
            elif iterazioni > 1000 and force: # richiesta dell'utente di forzare comunque l'esplorazione esaustiva
                print("\nForzo la ricerca degli iper-parametri ottimali.")
                print("N. iterazioni: %d "%iterazioni)
                print("Questo potrebbe richiedere un po' di tempo...")
            else:
                print("\nRandom search: %d iterazioni per %d combinazioni di parametri" % (iterazioni,n_iter))
            search = RandomizedSearchCV(
                self.model,
                params_dict,
                refit=True,
                n_iter=n_iter,
                cv=folds,
                random_state=random_state,
                n_jobs=n_jobs, # Parallelizzazione del processo di ricerca su n_jobs core (se disponibili)
                scoring=scoring,
                verbose=verbose
            )
        else: # ricerca esaustiva
            combinazioni = np.prod([len(params_dict[key]) for key in params_dict.keys()])
            iterazioni = n_splits * combinazioni
            if iterazioni > 1000 and not force: 
                raise ValueError("Troppe iterazioni (%d)! Prova a ridurre il numero di combinazioni di parametri o di partizioni da convalidare."%iterazioni)
            elif iterazioni > 1000 and force:
                print("\nForzo la ricerca degli iper-parametri ottimali.")
                print("N. iterazioni: %d "%iterazioni)
                print("Questo potrebbe richiedere un po' di tempo...")
            else:
                print("\nGrid search: %d iterazioni per %d combinazioni di parametri" % (iterazioni,combinazioni))
            search = GridSearchCV(
                self.model,
                params_dict,
                refit=True,
                cv=folds,
                n_jobs=n_jobs, # Parallelizzazione del processo di ricerca su n_jobs core (se disponibili)
                scoring=scoring,
                verbose=verbose
            )
        # ricerca dei parametri ottimali
        start = time()
        search.fit(X_train, y_train)
        results = pd.DataFrame(search.cv_results_).sort_values(by='rank_test_score')
        # stampa a video i risultati della ricerca in ordine di performance del modello con i parametri testati
        if verbose in [1,2,3]:
            print("\n")
            for i in range(0, len(results)):
                candidate = results.iloc[i]
                print(f"Model with rank: {candidate['rank_test_score']}")
                print(
                    "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                        float(abs(candidate['mean_test_score'])),
                        float(candidate['std_test_score']),
                    )
                )
                print("Parameters: {0}".format(candidate['params']))
                print("")

        if randomized:
                tipo = "casuale"
        else:
                tipo = "esaustiva"
        # stampa a video il modello che ha ottenuto il miglior punteggio medio su insieme di validazione
        # e i parametri che lo hanno generato
        print("\nLa ricerca %s ha impiegato: %.2f secondi (circa) per %d combinazioni di parametri\n" % (tipo,(time() - start),len(results)))
        print("MODELLO SELEZIONATO TRAMITE CV:")
        print("Punteggio medio su insime di validazione: {0:.3f} (std: {1:.3f})".format(
                        float(abs(results.iloc[1]['mean_test_score'])),
                        float(results.iloc[1]['std_test_score']),
                    )
                )
        print("Parametri selezionati: {0}".format(results.iloc[1]['params']))
        print("")

        self.set_ensemble(search.best_estimator_)
        return self

    
    def auto_features(self,
                      X,
                      threshold='median',
                      plot=True):
        """
        Funzione che implementa la selezione automatica delle feature più importanti per il modello.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vettore delle features.
        
        threshold : float, default='median'
            Soglia di selezione delle feature.
            Se 'median', la soglia viene impostata pari alla mediana delle importanze delle feature.
            Se 'mean', la soglia viene impostata pari alla media delle importanze delle feature.
            Altrimenti, la soglia viene impostata pari al valore passato.
        
        plot : bool, default=True
            Se True, viene visualizzata la barplot delle importanze delle feature.

        Returns
        -------
        sel_feat : list
            Lista delle feature selezionate.
        """
        # selezione semi-automatica del threshold per giudicare una feature importante
        if threshold == 'median':
            threshold = np.median(self.model.feature_importances_)
        elif threshold == 'mean':
            threshold = np.mean(self.model.feature_importances_)
        else: # threshold passato dall'utente
            threshold = threshold
        
        print(f"Threshold: {threshold}")
        fts_select = self.model.feature_importances_ > threshold 
        print(f"N. di feature selezionate: {sum(fts_select)}")
        print(f"Feature selezionate: {list(X.columns[fts_select])}")

        sel_feat = list(X.columns[fts_select])

        if plot:
            feature_imp = pd.DataFrame(
                    {"importance":self.model.feature_importances_},
                    index=list(X.columns)).sort_values("importance", ascending=False)
            fig, ax = subplots(figsize=(10, 6))
            ax.barh(feature_imp.index, feature_imp['importance'])
            if self.method == 'rf' and self.max_features is None:
                ax.set_title('Feature importance (Bagging)',loc='left')
            elif self.method == 'rf' and self.max_features is not None:
                ax.set_title('Feature importance (Random Forest)',loc='left')
            else:
                ax.set_title('Feature importance (Boosting)',loc='left')
            ax.set_xlabel('Importanza')
            ax.set_ylabel('Feature')
            ax.grid(True)
            ax.axvline(threshold, color='red', linestyle='--', label='Threshold')
            ax.legend()
            fig.tight_layout()
            plt.show()
        
        return sel_feat
