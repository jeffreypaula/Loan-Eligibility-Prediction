from sklearn.model_selection import cross_val_score, KFold,RepeatedKFold,RepeatedStratifiedKFold,StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from numpy import mean,sqrt,abs



def performance(func):
    def wrapper(*args,**kwargs):
       
    '''
        func: It computes the mean_score of model(s) on trainset using sklearn cross_validation functions.
        X: train dataset
        y: labels
        models : dict, an item
                model to train on dataset
        transformer: transformer object to link to each of the models in the models dictionary.
        shuffle : bool , shuffles data for crossvalidation
        metric : a function to use to evaluate model performance . Not yet supported.
        scoring : scoring metric to use to evaluate models
        fit_param: dict object
                    specified hyperparameters for each models to use when calculating the above metric score.
        cross_validate : default- 'kfold'
                         also accepts 'repeatstratified','repeat', 'stratified'
        n_splits: default - 10 . 
                  How many folds the data should be split into.
        n_repeats: default - 3.
                  How many times crossvalidation should be repeated on each kfold.
        random_state: default  - None
                    used by the random seed generator to generate random numbers.
    
    '''
        model_performance,scoring = func(*args,**kwargs)
        print(f'Scoring System : {scoring}')
        for model_name, performance in model_performance.items():
            print(f'{model_name} =  {performance}')

    return wrapper

def pipeline_model(transformer, models):
    
    '''
        func: it connects transformers to models using a pipeline.
    
    '''
    piped_models = {}
    for name,model in models.items():        
        piped_model=  Pipeline([('transformer', transformer),(name, model)])
        piped_models[name] = piped_model
    return piped_models



@performance
def spotcheck(X,y,models,transformer = None, shuffle=False,metric = None, scoring='neg_mean_squared_error',fit_param=None,cross_validate = 'kfold',
                        n_splits=10,n_repeats=3,random_state=None):  
    
    
    if cross_validate == 'repeat':
        kfold = RepeatedKFold(n_splits = n_splits, n_repeats = n_repeats, random_state=random_state)
    elif cross_validate == 'repeatstratified':
        kfold = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats=n_repeats, random_state=random_state)
    elif cross_validate == 'stratified':
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    else:
        kfold = KFold(n_splits= n_splits,shuffle=shuffle, random_state = random_state)
        
    model_performance = {}   
    
    if transformer:
        piped_models = pipeline_model(transformer, models)
        for name, model in piped_models.items():
            print(name)
            if fit_param:
                param = fit_param[name]
            else: 
                param = None
            score = cross_val_score(model, X,y,scoring= scoring,fit_params= param, cv= kfold,n_jobs=-1)
            model_performance[name] = mean(score)
    else:
        for name, model in models.items():
            if fit_param:
                param = fit_param[name]
            else:
                param = None
            score = cross_val_score(model, X,y,scoring= scoring,fit_params= param, cv= kfold,n_jobs=-1)
            model_performance[name] = abs(mean(score))
    
    return model_performance, scoring

    
    
    
    



#%%
