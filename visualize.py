import numpy as np
import pickle
from matplotlib import pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def main():
    MOVING_AVERAGE_K = 30

    with open('path-sgd.data', 'rb') as path_sgd_filename:
        path_sgd = pickle.load(path_sgd_filename)
        
        path_sgd_tr_err_curve = moving_average(path_sgd[0], MOVING_AVERAGE_K)
        path_sgd_tr_loss_curve = moving_average(path_sgd[1], MOVING_AVERAGE_K)
        path_sgd_val_err_curve = moving_average(path_sgd[2], MOVING_AVERAGE_K)
        path_sgd_val_loss_curve = moving_average(path_sgd[3], MOVING_AVERAGE_K)
    
    with open('sgd.data', 'rb') as sgd_filename:
        sgd = pickle.load(sgd_filename)
        
        sgd_tr_err_curve = moving_average(sgd[0], MOVING_AVERAGE_K)
        sgd_tr_loss_curve = moving_average(sgd[1], MOVING_AVERAGE_K)
        sgd_val_err_curve = moving_average(sgd[2], MOVING_AVERAGE_K)
        sgd_val_loss_curve = moving_average(sgd[3], MOVING_AVERAGE_K)
    
    with open('adagrad.data', 'rb') as adagrad_filename:
        adagrad = pickle.load(adagrad_filename)
        
        adagrad_tr_err_curve = moving_average(adagrad[0], MOVING_AVERAGE_K)
        adagrad_tr_loss_curve = moving_average(adagrad[1], MOVING_AVERAGE_K)
        adagrad_val_err_curve = moving_average(adagrad[2], MOVING_AVERAGE_K)
        adagrad_val_loss_curve = moving_average(adagrad[3], MOVING_AVERAGE_K)
    
    with open('path-sgd-u.data', 'rb') as path_sgd_u_filename:
        path_sgd_u = pickle.load(path_sgd_u_filename)
        
        path_sgd_u_tr_err_curve = moving_average(path_sgd_u[0], MOVING_AVERAGE_K)
        path_sgd_u_tr_loss_curve = moving_average(path_sgd_u[1], MOVING_AVERAGE_K)
        path_sgd_u_val_err_curve = moving_average(path_sgd_u[2], MOVING_AVERAGE_K)
        path_sgd_u_val_loss_curve = moving_average(path_sgd_u[3], MOVING_AVERAGE_K)
    
    with open('sgd-u.data', 'rb') as sgd_u_filename:
        sgd_u = pickle.load(sgd_u_filename)
        
        sgd_u_tr_err_curve = moving_average(sgd_u[0], MOVING_AVERAGE_K)
        sgd_u_tr_loss_curve = moving_average(sgd_u[1], MOVING_AVERAGE_K)
        sgd_u_val_err_curve = moving_average(sgd_u[2], MOVING_AVERAGE_K)
        sgd_u_val_loss_curve = moving_average(sgd_u[3], MOVING_AVERAGE_K)
    
    with open('adagrad-u.data', 'rb') as adagrad_u_filename:
        adagrad_u = pickle.load(adagrad_u_filename)
        
        adagrad_u_tr_err_curve = moving_average(adagrad_u[0], MOVING_AVERAGE_K)
        adagrad_u_tr_loss_curve = moving_average(adagrad_u[1], MOVING_AVERAGE_K)
        adagrad_u_val_err_curve = moving_average(adagrad_u[2], MOVING_AVERAGE_K)
        adagrad_u_val_loss_curve = moving_average(adagrad_u[3], MOVING_AVERAGE_K)

    plt.figure('Training loss')
    plt.title('Training loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(top = 2.5)
    adagrad_handle, = plt.plot(adagrad_tr_loss_curve, label = 'adagrad - balanced', color = 'black')
    path_sgd_handle, = plt.plot(path_sgd_tr_loss_curve, label = 'path-sgd', color = 'blue')
    sgd_handle, = plt.plot(sgd_tr_loss_curve, label = 'sgd - balanced', color = 'red')
    adagrad_u_handle, = plt.plot(adagrad_u_tr_loss_curve, label = 'adagrad - unbalanced', color = 'black', linestyle = 'dashed')
    path_sgd_u_handle, = plt.plot(path_sgd_u_tr_loss_curve, label = 'path-sgd - unbalanced', color = 'blue', linestyle = 'dashed')
    sgd_u_handle, = plt.plot(sgd_u_tr_loss_curve, label = 'sgd - unbalanced', color = 'red', linestyle = 'dashed')
    plt.legend(handles=[adagrad_handle, path_sgd_handle, sgd_handle, adagrad_u_handle, path_sgd_u_handle, sgd_u_handle])
    plt.show()

    plt.figure('Train error')
    plt.title('Train error')
    plt.xlabel('epochs')
    plt.ylabel('error rate')
    plt.ylim(top = 0.05)
    adagrad_handle, = plt.plot(adagrad_tr_err_curve, label = 'adagrad - balanced', color = 'black')
    path_sgd_handle, = plt.plot(path_sgd_tr_err_curve, label = 'path-sgd', color = 'blue')
    sgd_handle, = plt.plot(sgd_tr_err_curve, label = 'sgd - balanced', color = 'red')
    adagrad_u_handle, = plt.plot(adagrad_u_tr_err_curve, label = 'adagrad - unbalanced', color = 'black', linestyle = 'dashed')
    path_sgd_u_handle, = plt.plot(path_sgd_u_tr_err_curve, label = 'path-sgd - unbalanced', color = 'blue', linestyle = 'dashed')
    sgd_u_handle, = plt.plot(sgd_u_tr_err_curve, label = 'sgd - unbalanced', color = 'red', linestyle = 'dashed')
    plt.legend(handles=[adagrad_handle, path_sgd_handle, sgd_handle, adagrad_u_handle, path_sgd_u_handle, sgd_u_handle])
    plt.show()

    plt.figure('Val error')
    plt.title('Val error')
    plt.xlabel('epochs')
    plt.ylabel('error rate')
    plt.ylim(top = 0.1)
    adagrad_handle, = plt.plot(adagrad_val_err_curve, label = 'adagrad - balanced', color = 'black')
    path_sgd_handle, = plt.plot(path_sgd_val_err_curve, label = 'path-sgd', color = 'blue')
    sgd_handle, = plt.plot(sgd_val_err_curve, label = 'sgd - balanced', color = 'red')
    adagrad_u_handle, = plt.plot(adagrad_u_val_err_curve, label = 'adagrad - unbalanced', color = 'black', linestyle = 'dashed')
    path_sgd_u_handle, = plt.plot(path_sgd_u_val_err_curve, label = 'path-sgd - unbalanced', color = 'blue', linestyle = 'dashed')
    sgd_u_handle, = plt.plot(sgd_u_val_err_curve, label = 'sgd - unbalanced', color = 'red', linestyle = 'dashed')
    plt.legend(handles=[adagrad_handle, path_sgd_handle, sgd_handle, adagrad_u_handle, path_sgd_u_handle, sgd_u_handle])
    plt.show()

if __name__ == '__main__':
    main()