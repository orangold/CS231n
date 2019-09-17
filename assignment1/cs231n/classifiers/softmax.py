from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)
        scores-=np.max(scores) # This is a stability trick, as exponents can get abnormally large.
        scores_sum = np.sum(np.exp(scores))
        loss-= scores[y[i]] - np.log(scores_sum)
        probabilities_score = np.exp(scores) / np.sum(np.exp(scores))
        probabilities_score[y[i]]-=1
        dW+=np.dot(X[i][:,np.newaxis],probabilities_score[np.newaxis,:])

    loss+=reg*np.sum(W*W)
    loss/=num_train
    dW+=reg*2*W
    dW/=num_train   

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores=np.dot(X,W)
    scores-=np.max(scores,axis=1)[:,np.newaxis]
    probabilities_sum = np.sum(np.exp(scores),axis=1)
    loss=np.sum(scores[np.arange(scores.shape[0]),y] - np.log(probabilities_sum))
    probabilities_score = np.exp(scores) / np.sum(np.exp(scores),axis=1)[:,np.newaxis]
    probabilities_score[np.arange(scores.shape[0]),y]-=1
    dW=np.dot(X.T,probabilities_score)
    loss+=reg*np.sum(W*W)
    loss/=num_train
    dW+=reg*2*W
    dW/=num_train  
    #copy paste from http://cs231n.github.io/neural-networks-case-study/#loss
    score = X.dot(W) # (N,C)
    score = score - np.amax(score,axis = 1,keepdims = True)

    score = np.exp(score) 

    probs = score/np.sum(score,axis = 1, keepdims = True)

    loss = -1*np.log(probs[np.arange(num_train),y]).sum()/num_train

    loss = loss + 0.5 * reg * np.sum(W * W)

  #http://cs231n.github.io/neural-networks-case-study/#grad

    dscores = probs #(N,C)
    dscores[range(num_train),y] -= 1
    dscores = dscores / num_train
    dW = np.dot(X.T,dscores)
    dW += reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
