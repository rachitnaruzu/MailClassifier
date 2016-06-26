import numpy as np

def sigmoid(z):
    e = 2.71828182846;
    g = 1 / (1 + e ** (-1 * z))
    return g

def predict(theta,X):
    res = np.dot(X,theta)
    return res >= 0

def get_accuracy(res,Y):
    return np.mean((res == Y) * 100)
    
def cost_function(theta,X,Y,lmbda):
    h = sigmoid(np.dot(X,theta))
    m,n = X.shape
    costi = -(Y*np.log(h) + (1-Y)*np.log(1-h))
    thetad = np.delete(theta,0,0)
    thetad = thetad.reshape(n-1,1)
    J = (costi.sum())/m + np.dot(thetad.T,thetad)*(lmbda/(2*m))
    return J
    

def train(X,Y):
    
    m,n = X.shape
    
    theta = np.zeros(shape=(n,1))
    
    lmbda = 0
    alpha = 0.5
    
    num_of_iter = 2000
    
    J_history = np.zeros(shape = (num_of_iter,1))
    
    temp = np.zeros(shape=theta.shape)
    it = 0;
    while(True and it < num_of_iter ):
        h = sigmoid(np.dot(X,theta))
        
        temp[0] = theta[0] - alpha*(np.dot((h - Y).T,X[:,[0]])/m)
        for i in range(1,n):
            temp[i] = theta[i] - alpha*(np.dot((h - Y).T,X[:,[i]])/m + (theta[i] * lmbda / m))
            
        theta = temp
        
        J_history[it][0] = cost_function(theta, X, Y, lmbda)
        #print(it,J_history[it][0])
        
        if(it != 0 and (J_history[it-1][0] - J_history[it][0] < 0.00001)):
            break
        
        it += 1
        
    
    res = predict(theta,X)
    
    accuracy = get_accuracy(res, Y)
    
    return theta,accuracy,J_history,it




#input_layer_size  = n;


'''
costi = np.zeros(shape=(m,1))
one = np.where(Y == 1)[0]
zero = np.where(Y == 0)[0]
costi[one] = -np.log(h[one])
costi[zero] = -np.log(1 - h[zero])
'''
