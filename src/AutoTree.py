from src.utils import *
# Classe AutoTree ###########################################################################################################################################################
class AutoTree:
    """
    Class that allows the creation of a regression or classification tree
    with default or custom parameters.
    """
    # costruttore
    def __init__(self, 
                 obj='regressione', 
                 criterion='squared_error', 
                 max_depth=5, 
                 min_samples_split=2, 
                 min_samples_leaf=5, 
                 random_state=0, 
                 max_leaf_nodes=None, 
                 min_impurity_decrease=0.002, 
                 ccp_alpha=0.0,
                 tree = None):
        """
    Parameters
    ----------
    obj : str, default='regression'
        The type of tree to create. Can be 'regression' or 'classification'.

    criterion : str, default='squared_error'
        The measure of quality of a split used for selecting the feature.
        For regression can be `"squared_error", "friedman_mse", "absolute_error" or "poisson"`
        For classification can be `"gini", "entropy", "log_loss"` (default is "gini").
    
    max_depth : int, default=5
        The maximum depth of the tree. If None, nodes are expanded until
        all leaves are pure or until all leaves contain less than
        `min_samples_split` samples.
        In this implementation the maximum depth is quite generous 
        (given the binary structure we are implicitly allowing a maximum of 25 leaf nodes), 
        however, the value `None` is not left as default because
        it would almost certainly lead to an overfitting of the model to the data.
    
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node.
        If `int`, considers min_samples_split as the minimum number.
        If `float`, min_samples_split is a fraction and `ceil(min_samples_split * n_samples)`
        are the minimum samples for each split.
    
    min_samples_leaf : int or float, default=5
        The minimum number of samples required to be at a leaf node.
        If `int`, considers min_samples_leaf as the minimum number.
        If `float`, min_samples_leaf is a fraction and `ceil(min_samples_leaf * n_samples)`
        are the minimum samples for each leaf node.
    
    random_state : int, RandomState instance or None, default=0
        If `int`, random_state is the seed used by the random number generator.
        If `RandomState` instance, random_state is the random number generator.
        If `None`, the random number generator is the RandomState instance used by np.random.
    
    max_leaf_nodes : int, default=None
        The maximum number of leaves. If `None`, the number of leaves is not limited.
    
    min_impurity_decrease : float, default=0.002
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
    
    ccp_alpha : non-negative float, default=0.0
        The complexity parameter used for Minimal Cost-Complexity Pruning. The greater the value of ccp_alpha,
        the more nodes are pruned. If set to 0, no pruning is performed.
"""

        self.obj = obj
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.tree = tree

        # Albero di REGRESSIONE
        if obj == 'regressione':
            self.tree = DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                ccp_alpha=self.ccp_alpha
            )
        else: # Albero di CLASSIFICAZIONE
            self.obj = 'classificatione'
            if self.criterion.lower() not in ['gini', 'entropy', 'log_loss']:
                self.criterion = 'gini'
            self.tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                ccp_alpha=self.ccp_alpha
            )
        print(f"{'#'*5} Albero di {self.obj.capitalize()} creato con successo! {'#'*5}")

    def __repr__(self):
        """
        Method that establishes the string representation of the object.
        """
        if self.tree is None: 
            return f"\nTree of {self.obj.capitalize()},\
                \n> Leaf impurity criterion used: {self.criterion.upper()}.\
                \n> Maximum depth: {self.max_depth}.\
                    \n> Min. number of samples required to split an internal node: {self.min_samples_split}.\
                    \n> Min. number of samples required to be at a leaf node: {self.min_samples_leaf}.\
                        \n> Seed used by the random number generator: {self.random_state}.\
                        \n> Max. number of leaves: {self.max_leaf_nodes}.\
                            \n> A node will be split if this split induces a decrease of impurity greater than or equal to: {self.min_impurity_decrease}.\
                            \n> Complexity parameter used for pruning: {round(self.ccp_alpha,3)} "
        else:
            return f"\nTree of {self.obj.capitalize()},\
                \n> Leaf impurity criterion used: {self.criterion.upper()}.\
                \n> Maximum depth: {self.tree.max_depth}.\
                    \n> Min. number of samples required to split an internal node: {self.tree.min_samples_split}.\
                    \n> Min. number of samples required to be at a leaf node: {self.tree.min_samples_leaf}.\
                        \n> Seed used by the random number generator: {self.tree.random_state}.\
                        \n> Max. number of leaves: {self.tree.max_leaf_nodes}.\
                            \n> A node will be split if this split induces a decrease of impurity greater than or equal to: {self.tree.min_impurity_decrease}.\
                            \n> Complexity parameter used for pruning: {round(self.tree.ccp_alpha,3)} "

    def set_tree(self, tree):
        """
        Method to set an already created tree.

        Parameters
        ----------
        tree : DecisionTreeRegressor or DecisionTreeClassifier
            The tree to be set in our object.
            
        Returns
        -------
        None
        """
        self.tree = tree
        self.criterion = tree.criterion
        self.max_depth = tree.max_depth
        self.min_samples_split = tree.min_samples_split
        self.min_samples_leaf = tree.min_samples_leaf
        self.random_state = tree.random_state
        self.max_leaf_nodes = tree.max_leaf_nodes
        self.min_impurity_decrease = tree.min_impurity_decrease
        self.ccp_alpha = tree.ccp_alpha

    
    def fit(self, X, y):
        """
        Method to fit the tree to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target vector.
        
        Returns
        -------
        None
        """
        self.tree.fit(X, y)
    
    def predict(self, X):
        """
        Method to make predictions on the samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        
        Returns
        -------
        array-like of shape (n_samples,) or (n_samples, n_outputs)
            Prediction vector.
        """

        return self.tree.predict(X)
    
    def score(self, X, y):
        """
        Method to calculate the accuracy of the model
        (classification and regression).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature/covariate matrix.
        
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target vector.
        
        Returns
        -------
        float
            Accuracy of the model.
        """
        return self.tree.score(X, y)

    def auto_fit(self,
                    X,
                    y,
                    verbose = 1,
                    plot = False):
        """
        Method to fit the tree to data and to display fitting statistics,
        decision rules, and the average cross-validation score.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Response vector.
        
        random_state : int, default=0
            Seed used by the random number generator.
        
        verbose : int, default=1
            If 0, prints nothing.
            If 1, prints fitting statistics.
            If 2, prints fitting statistics and the `importance` level for each covariate.
            If 3, prints fitting statistics, the `importance` level for each covariate, and the decision rules.
        
        plot : bool, default=False
            If True, plots the tree.
        
        Returns
        -------
        None
        """
            
        features = list(X.columns)
        start = time() 
        self.fit(X,y) 
        # Informazioni di adattamento
        if verbose in [1,2,3]:
            print('\nFIT STATS:')
            print('---------------------------------')
            print(f"> Tree depth: {self.tree.get_depth()}")
            print(f"> Number of terminal nodes (leaves): {self.tree.get_n_leaves()}")
            print(f"Time taken for fitting:", round(time() - start, 3), 's')

            if verbose in [2,3]:
                # importanza delle covariate
                importance = pd.DataFrame(
                    {"Importance":self.tree.feature_importances_,
                     "Variables":list(X.columns)}).sort_values(by='Importance',ascending=False)[["Variables","Importance"]]
                print('\nFEATURE IMPORTANCE:')
                print('----------------------------------')
                for i in range(len(importance)): 
                    print(f'Feature {importance.iloc[i,0]}: Importance score = {importance.iloc[i,1]:.3f}')
                if verbose == 3:
                    print('\nDECISION RULES:')
                    print('----------------------------------')
                    print(
                        export_text(
                        self.tree,
                        feature_names=features,
                        show_weights=True)
                        )
        # grafico dell'albero decisionale, con numero di osservazioni per nodo, 
        # valore medio della risposta per nodo e indice di impurita del nodo
        if plot:
            ax = subplots(figsize=(20, 20),layout='constrained')[1]
            plot_tree(self.tree, 
                    filled=True,
                    label='all',
                    ax=ax,
                    feature_names= features,
                    rounded=True, 
                    precision=2)
            plt.title(f"Albero di {self.obj.capitalize()} - Indice: {self.tree.criterion.upper()} - CP: {round(self.tree.ccp_alpha,3)}", fontsize=20)
            plt.show()
    

    def auto_prune(self,
                   X_train,
                   y_train,
                   random_state =0, 
                   n_splits=10,
                   plot=True,
                   print_path = False,
                   auto = True):
        """
    Method that prunes the tree to reduce the generalization error.
    To do this, the tree's cost complexity pruning path is computed (via cross-validation),
    and the alpha value (complexity parameter) that minimizes the generalization error is selected.
    (For more information: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html)

    The best alpha value is computed using cross-validation on the training set.
    We take the training set because pruning is to be considered part of the learning process.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Feature matrix (training set).

    y_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Response vector (training set).

    random_state : int, default=0
        Seed used by the random number generator.

    n_splits : int, default=10
        The number of folds used for cross-validation.

    plot : bool, default=True
        If True, prints the tree's pruning path, i.e., the total impurity of the leaves
        as a function of the complexity parameter alpha, and the model's accuracy for each alpha value.

    print_path : bool, default=False
        If True, prints the tree's pruning path (values of complexity alpha and total impurity of leaves).

    auto : bool, default=True
        If `False`, it requires the user to select the estimator's parameters manually.
        Selection is done by entering the number of leaves corresponding to the tree obtained from pruning
        with a certain CP value.

    Returns
    -------
    AutoTree
"""
        
        # calcolo del percoso di potatura dell'albero
        path = self.tree.cost_complexity_pruning_path(X_train, y_train)
        alphas = path.ccp_alphas
        
        if print_path:
            print(pd.DataFrame(path))        
        
        # selezioniamo il criterio di valutazione in base al metodo utilizzato (regressione o classificazione)
        if self.obj == 'regressione':
            scoring = 'neg_mean_absolute_percentage_error'
        else:
            scoring = 'accuracy'
        
        # calcolo dell'accuratezza del modello per ogni valore di alpha:
        kfold = KFold(n_splits = n_splits,
                  shuffle=True,
                  random_state=random_state)
        
        grid = GridSearchCV(self.tree,
                                param_grid={'ccp_alpha':alphas},
                                refit=True,
                                cv = kfold,
                                scoring=scoring)
        
        grid.fit(X_train, y_train)
        # cresce alpha, scende il N_leaves
        results = pd.DataFrame(grid.cv_results_)[
            ['param_ccp_alpha',
             'mean_test_score',
             'std_test_score',
             'rank_test_score']
             ]
        results['N_foglie'] = np.arange(1,len(results)+1)[::-1]
        results['mean_test_score'] = abs(results['mean_test_score'])
        
        if plot:
            # percorso di potatura dell'albero: alpha vs impurità totale
            axs = plt.subplots(nrows=2, figsize=(9,8),layout='constrained' ) [1]
            axs[0].plot(alphas,
                        results.N_foglie, 
                        marker="o", 
                        linestyle="-",
                        drawstyle="steps-post",
                        color="black")
            axs[0].set_xlabel("Alpha")
            axs[0].set_ylabel("Numero Foglie")
            axs[0].set_title("Percorso di potatura dell'albero")
            # risultati convalida incrociata: alpha vs performance
            axs[1].set_xlabel("CP")
            axs[1].set_ylabel("X-val Error")
            axs[1].set_title("Error vs CP")
            # plot the test scores with error bars
            axs[1].errorbar(results.N_foglie, 
                            results.mean_test_score, 
                            yerr=results.std_test_score,
                            fmt="o", 
                            markerfacecolor='none',
                            label="test", 
                            linestyle="-",
                            drawstyle="default",
                            color="black",
                            ecolor='red',
                            capsize=2,  
                            elinewidth=0.5)
            axs[1].set_xticks(results.N_foglie.astype(int))
            ax2 = axs[1].twiny()
            ax2.set_xticklabels([round(x,2) for i,x in enumerate(alphas[::-1]) if i%3==0])
            axs[1].tick_params(axis='x',top=True)
            axs[1].legend()
            # retta dell'errore minimo: gli errori sotto la retta sono approssimativamente sotto la retta.
            err_min = results.mean_test_score.min()
            std_err_min = results.iloc[results.mean_test_score.idxmin(),2]
            axs[1].axhline(y=(err_min + std_err_min),linestyle =':',color="black")
            plt.show()
        # Selezione manuale del valore di alpha
        if not auto:
            print('\nCROSS-VALIDATION RISULTS:')
            print('---------------------------------')
            print(results)
            best_selected_tree = results[
                results['N_foglie']==int(
                input("N. foglie albero selezionato per potatura:"))]
            selected_cp = float(best_selected_tree['param_ccp_alpha'])
            print(f"Il valore CP corrispondente all'albero con il numero di foglie selezionato è: {round(selected_cp,3)}")
            self.tree.set_params(ccp_alpha=selected_cp)
        else:
            auto_cp = grid.best_params_['ccp_alpha']
            print(f"Il valore CP selezionato automaticamente è: {round(auto_cp,3)}")
            bestTree = grid.best_estimator_
            self.set_tree(bestTree)
        
        return self
