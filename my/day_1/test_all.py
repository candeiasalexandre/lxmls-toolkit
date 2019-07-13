import lxmls.classifiers.perceptron as percc
import lxmls.classifiers.mira as mirac
import lxmls.classifiers.max_ent_batch as mebc
import lxmls.classifiers.max_ent_online as meoc
import lxmls.classifiers.svm as svmc
import lxmls.classifiers.naive_bayes as nbc

import ipdb

def test_all(x_train, y_train, x_test, y_test, hyperparameter_values):
    models_params_result = {}
    models_acc_train = {}
    models_acc_test = {}
    

    models = {"perceptron":percc.Perceptron(), "naivebayes":nbc.NaiveBayes(), "mira":mirac.Mira(), "me_lbfgs":mebc.MaxEntBatch(), "me_sgd":meoc.MaxEntOnline(), "svm":svmc.SVM()}

    for key, model in models.items():
        models_params_result[key] = []
        models_acc_train[key] = []
        models_acc_test[key] = []
        if key == "perceptron" or key=="naivebayes":
            params_model = model.train(x_train, y_train)
            models_params_result[key].append(params_model)
            
            y_pred_train = model.test(x_train, params_model)
            acc_train = model.evaluate(y_train, y_pred_train)
            #ipdb.set_trace()
            models_acc_train[key].append(acc_train)

            y_pred_test = model.test(x_test, params_model)
            acc_test = model.evaluate(y_test, y_pred_test)
            models_acc_test[key].append(acc_test)

            print("{} Dataset train: {} test: {}".format(key, acc_train, acc_test))
        else:
            for hyperparameter in hyperparameter_values:
                model.regularizer = hyperparameter # This is lambda
                params_model = model.train(x_train, y_train)
                models_params_result[key].append(params_model)

                y_pred_train = model.test(x_train, params_model)
                acc_train = model.evaluate(y_train, y_pred_train)
                #ipdb.set_trace()
                models_acc_train[key].append(acc_train)

                y_pred_test = model.test(x_test, params_model)
                acc_test = model.evaluate(y_test, y_pred_test)
                models_acc_test[key].append(acc_test)

                print("{} Dataset lambda: {} train: {} test: {}".format(key, hyperparameter, acc_train, acc_test))

    return models_params_result, models_acc_train, models_acc_test

import lxmls.readers.simple_data_set as sds
import numpy as np
if __name__ == "__main__":
    sd = sds.SimpleDataSet(
        nr_examples=100,
        g1=[[-1,-1],2], 
        g2=[[1,1],1], 
        balance=0.1,
        split=[0.7,0,0.3] )
        
    hyperparameter_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    models_params, models_acc_train, models_acc_test = test_all(sd.train_X, sd.train_y, sd.test_X, sd.test_y, hyperparameter_values)
    
    fig, axis = sd.plot_data("osx")
    for key, value in models_params.items():
        for idx in range(len(models_acc_train[key])):
            best_hyperparameter = np.argmax(models_acc_test)
            best_model = models_params[key][best_hyperparameter]
            fig, axis = sd.add_line(fig, axis, best_model, key, np.random.rand(3,))
            print("{} Model with Hyperparam: {}, train_acc: {}, test_acc {}".format(key, hyperparameter_values[idx], models_acc_train[key][idx], models_acc_test[key][idx]))
            