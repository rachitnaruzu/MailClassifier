import numpy as np

def sigmoid(z):
    e = 2.71828182846
    g = 1 / (1 + e ** (-1 * z))
    return g

def predict(Theta1,Theta2,X):
    m,_ = X.shape
    z1 = np.dot(Theta1,X.T)
    a1 = sigmoid(z1).T
    a1 = np.concatenate([np.ones(shape=(m,1)),a1],axis=1)
    z2 = np.dot(Theta2,a1.T)
    h = sigmoid(z2).T
    return h >= 0.5

def get_accuracy(res,Y):
    return np.mean((res == Y) * 100)

def cost_function(Theta1,Theta2, h, Y, lmbda):
    m,_ = h.shape
    
    cost = Y*np.log(h) + (1-Y)*np.log(1-h)
    Theta1d = np.delete(Theta1,0,1)
    Theta2d = np.delete(Theta2,0,1)
    J = -cost.sum()/m + ((Theta1d ** 2).sum() + (Theta2d ** 2).sum()) * (lmbda/(2*m))
    
    return J

def grad(nnparams,X,Y,lmbda,hidden_layer_size,output_layer_size):
    m,n = X.shape
    
    Theta1 = np.reshape(nnparams[0:hidden_layer_size*n],(hidden_layer_size,n))
    Theta2 = np.reshape(nnparams[hidden_layer_size*n:np.size(nnparams)],(1,hidden_layer_size+1))
    
    z1 = np.dot(Theta1,X.T)
    a1 = sigmoid(z1).T
    a1 = np.concatenate([np.ones(shape=(m,1)),a1],axis=1)
    z2 = np.dot(Theta2,a1.T)
    h = sigmoid(z2).T
    
    Theta2d = np.delete(Theta2,0,1)
    
    delFinal = (h - Y).T
    zg = sigmoid(z1)
    delHidden = np.dot(Theta2d.T,delFinal) * zg* (1 - zg)
    
    Delta2 = np.dot(delFinal,a1)
    Delta1 = np.dot(delHidden,X)
    
    Temp2 = np.concatenate([np.zeros(shape=(output_layer_size,1)),np.delete(Theta2,0,1)],axis=1);
    Temp1 = np.concatenate([np.zeros(shape=(hidden_layer_size,1)),np.delete(Theta1,0,1)],axis=1);
    
    Theta2_grad = Delta2 + Temp2 *lmbda;
    Theta1_grad = Delta1 + Temp1 *lmbda;
    
    Theta2_grad = Theta2_grad / m;
    Theta1_grad = Theta1_grad / m;
    
    Theta1_grad_unrolled = np.reshape(Theta1_grad,np.size(Theta1_grad))
    Theta2_grad_unrolled = np.reshape(Theta2_grad,np.size(Theta2_grad))
    
    gradparams = np.append(Theta1_grad_unrolled,Theta2_grad_unrolled)
    
    return gradparams
    

def train(X,Y):
    m,n = X.shape
    
    hidden_layer_size = 5
    output_layer_size = 1
    
    Theta1 = np.random.rand(hidden_layer_size,n)
    Theta2 = np.random.rand(output_layer_size,hidden_layer_size+1)
    
    '''
    Theta1 = np.loadtxt("Theta1.csv",delimiter=",")
    Theta2 = np.loadtxt("Theta2.csv",delimiter=",")
    Theta2 = np.array([Theta2])
    '''
    
    lmbda = 4.5
    alpha = 1
    
    num_of_iter = 3000
    
    J_history = np.zeros(shape = (num_of_iter,1))
    
    it = 0;
    while(True and it < num_of_iter):
    
        z1 = np.dot(Theta1,X.T)
        a1 = sigmoid(z1).T
        a1 = np.concatenate([np.ones(shape=(m,1)),a1],axis=1)
        z2 = np.dot(Theta2,a1.T)
        h = sigmoid(z2).T
        
        Theta2d = np.delete(Theta2,0,1)
        
        delFinal = (h - Y).T
        zg = sigmoid(z1)
        delHidden = np.dot(Theta2d.T,delFinal) * zg * (1 - zg)
        
        Delta2 = np.dot(delFinal,a1)
        Delta1 = np.dot(delHidden,X)
        
        Temp2 = np.concatenate([np.zeros(shape=(1,1)),np.delete(Theta2,0,1)],axis=1);
        Temp1 = np.concatenate([np.zeros(shape=(hidden_layer_size,1)),np.delete(Theta1,0,1)],axis=1);
        
        Theta2_grad = Delta2 + Temp2 *lmbda;
        Theta1_grad = Delta1 + Temp1 *lmbda;
        
        Theta2_grad = Theta2_grad / m;
        Theta1_grad = Theta1_grad / m;
        
        Theta2 = Theta2 - alpha * Theta2_grad
        Theta1 = Theta1 - alpha * Theta1_grad
        
        J_history[it][0] = cost_function(Theta1,Theta2, h, Y, lmbda)
        
        #print(it,J_history[it][0])
        #print(a1)
        #print(delFinal)
        #print(Delta2)
        
        if(it != 0 and (J_history[it-1][0] - J_history[it][0] < 0.00000001)):
            break
        
        it += 1
        
    
    res = predict(Theta1,Theta2,X)
    
    accuracy = get_accuracy(res, Y)
    
    return Theta1,Theta2,accuracy,J_history,it
    
