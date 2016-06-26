import create_data as cd
import logistic_regression as lr
import numpy as np
import time
import matplotlib.pyplot as plt
import artificial_neural_network as ann
import os
from sklearn import svm


def plot_J_history(it,J_history,title):
    plt.xlabel('Iterations')
    plt.ylabel('Cost (J)')
    plt.plot([i for i in range(1,it+1)],[J_history[i][0] for i in range(0,it)])
    plt.suptitle(title)
    plt.draw()

def run_logistic_regression(Train_X,Train_Y,Test_X,Test_Y):
    
    print ('Logistic_Regression:')
    
    print ('\nTraining begins...')
    m,_ = Train_X.shape
    Train_X = np.concatenate([np.ones(shape=(m,1)),Train_X],axis=1)
    start_time = time.time()
    theta,accuracy,J_history,it = lr.train(Train_X,Train_Y)
    end_time = time.time()
    print('iterations: {it}'.format(it = it));
    print('time_taken: {time} sec'.format(time = (end_time - start_time)))
    print('accuracy: {acc}'.format(acc = accuracy))
    
    plot_J_history(it,J_history,'Logistic Regression')
    
    print ('\nValidation on test data:')
    m,_ = Test_X.shape
    Test_X = np.concatenate([np.ones(shape=(m,1)),Test_X],axis=1)
    res = lr.predict(theta,Test_X)
    accuracy = lr.get_accuracy(res, Test_Y)
    print('validation accuracy: {acc}'.format(acc = accuracy))
    
    print ('\nTop 30 Spam Predictors:')
    features = cd.get_features()
    feature_w = zip(features, theta[1:,0])
    sorted_feature_w = sorted(feature_w,key= lambda t:(t[1],t[0]),reverse=True)
    for i in range(30):
        print ('%s\t\t%.2f' % (sorted_feature_w[i][0], sorted_feature_w[i][1]))
    
    print ('\n\n')
    
def run_ann(Train_X,Train_Y,Test_X,Test_Y):
    
    print ('Feed_Forward_ANN:')
    
    print ('\nTraining begins...')
    m,_ = Train_X.shape
    Train_X = np.concatenate([np.ones(shape=(m,1)),Train_X],axis=1)
    start_time = time.time()
    Theta1,Theta2,accuracy,J_history,it = ann.train(Train_X,Train_Y)
    end_time = time.time()
    print('iterations: {it}'.format(it = it));
    print('time_taken: {time} sec'.format(time = (end_time - start_time)))
    print('accuracy: {acc}'.format(acc = accuracy))
    
    plot_J_history(it,J_history,'Feed_Forward_ANN')
    
    print ('\nValidation on test data:')
    m,_ = Test_X.shape
    Test_X = np.concatenate([np.ones(shape=(m,1)),Test_X],axis=1)
    res = ann.predict(Theta1,Theta2,Test_X)
    accuracy = ann.get_accuracy(res, Test_Y)
    print('validation accuracy: {acc}'.format(acc = accuracy))
    
    print ('\nTop 30 Spam Predictors:')
    features = cd.get_features()
    w = Theta2[:,1:].T
    weights = np.dot(Theta1.T, w)
    weights = weights.T
    weights = weights[0,1:]
    feature_w = zip(features, weights)
    sorted_feature_w = sorted(feature_w,key= lambda t:(t[1],t[0]),reverse=True)
    for i in range(30):
        print ('%s\t\t%.2f' % (sorted_feature_w[i][0], sorted_feature_w[i][1]))
    
    print ('\n\n')
    

def run_svm_linear(Train_X,Train_Y,Test_X,Test_Y):
    
    print ('Support Vector Machine (Linear):')
    
    svm_clf = svm.LinearSVC()
    
    print ('\nTraining begins...')
    start_time = time.time()
    svm_clf.fit(Train_X, Train_Y[:,0])
    end_time = time.time()
    res = svm_clf.predict(Train_X)
    accuracy = svm_clf.score(Train_X, Train_Y[:,0]) * 100
    print('time_taken: {time} sec'.format(time = (end_time - start_time)))
    print('accuracy: {acc}'.format(acc = accuracy))
    
    print ('\nValidation on test data:')
    res = svm_clf.predict(Test_X)
    accuracy = svm_clf.score(Test_X, Test_Y[:,0]) * 100
    print('validation accuracy: {acc}'.format(acc = accuracy))
    
    
    
    print ('\nTop 30 Spam Predictors:')
    features = cd.get_features()
    
    #print(len(features))
    #print(len(svm_clf.coef_[0,:]))
    
    weights = svm_clf.coef_[0,:]
    feature_w = zip(features, weights)
    sorted_feature_w = sorted(feature_w,key= lambda t:(t[1],t[0]),reverse=True)
    for i in range(30):
        print ('%s\t\t%.2f' % (sorted_feature_w[i][0], sorted_feature_w[i][1]))
    
    print ('\n\n')

    
def myrbfkernel(X, L):
    e = 2.71828182846
    sigma = 4
    xsq = np.sum(X ** 2, axis = 1)
    lsq = np.sum(L ** 2, axis = 1)
    k = -2 * np.dot(X, L.T)
    k = (k.T + xsq).T
    k = k + lsq
    k = k / (2 * sigma * sigma)
    k = e ** (-k)
    return k
    

def run_svm(Train_X,Train_Y,Test_X,Test_Y):
    
    print ('Support Vector Machine (RBF):')
    
    svm_clf = svm.SVC(C = 1, kernel = myrbfkernel)
    
    print ('\nTraining begins...')
    start_time = time.time()
    svm_clf.fit(Train_X, Train_Y[:,0])
    end_time = time.time()
    res = svm_clf.predict(Train_X)
    accuracy = svm_clf.score(Train_X, Train_Y[:,0]) * 100
    print('time_taken: {time} sec'.format(time = (end_time - start_time)))
    print('accuracy: {acc}'.format(acc = accuracy))
    
    #print(svm_clf.dual_coef_)
    
    print ('\nValidation on test data:')
    res = svm_clf.predict(Test_X)
    accuracy = svm_clf.score(Test_X, Test_Y[:,0]) * 100
    print('validation accuracy: {acc}'.format(acc = accuracy))
    
    print ('\n\n')

    
base = os.path.dirname(os.path.abspath('__file__')) + '/mails'
Train_X, Train_Y, Test_X, Test_Y = cd.get_data(base,split = 0.70, lengthfrac = 0.1)

#run_logistic_regression(Train_X, Train_Y, Test_X, Test_Y)
#run_ann(Train_X, Train_Y, Test_X, Test_Y)
run_svm_linear(Train_X, Train_Y, Test_X, Test_Y)
run_svm(Train_X, Train_Y, Test_X, Test_Y)

plt.show()

'''
features = ['about', 'above', 'account', 'act', 'activity', 'address', 'adobe', 'after', 'agreement', 'aimee', 'align', 'all', 'allen', 'also', 'am', 'america', 'ami', 'an', 'and', 'anita', 'any', 'aol', 'april', 'are', 'as', 'at', 'attached', 'available', 'back', 'based', 'be', 'because', 'been', 'before', 'being', 'below', 'best', 'bgcolor', 'biz', 'bob', 'body', 'border', 'both', 'br', 'brenda', 'brian', 'bryan', 'business', 'but', 'buy', 'buyback', 'by', 'call', 'calls', 'camp', 'can', 'cash', 'cc', 'cd', 'center', 'ces', 'cf', 'change', 'changes', 'charge', 'china', 'chokshi', 'cialis', 'click', 'clynes', 'color', 'com', 'companies', 'company', 'computron', 'contact', 'content', 'contract', 'contracts', 'corp', 'could', 'counterparty', 'country', 'cs', 'currently', 'daily', 'daren', 'darial', 'data', 'date', 'day', 'days', 'de', 'deal', 'deals', 'dec', 'delivery', 'desk', 'details', 'did', 'div', 'do', 'does', 'dollars', 'don', 'down', 'drugs', 'due', 'each', 'eastrans', 'easy', 'ect', 'effective', 'email', 'ena', 'energy', 'enron', 'entex', 'face', 'farmer', 'feb', 'february', 'file', 'first', 'flow', 'following', 'font', 'fontfont', 'for', 'forward', 'forwarded', 'free', 'friday', 'from', 'ftar', 'full', 'fund', 'future', 'fyi', 'gary', 'gas', 'generic', 'george', 'get', 'gif', 'give', 'global', 'go', 'gold', 'great', 'group', 'had', 'has', 'have', 'he', 'health', 'height', 'help', 'here', 'hi', 'high', 'his', 'home', 'hotlist', 'hou', 'how', 'howard', 'hpl', 'hplc', 'href', 'hsc', 'html', 'htmlimg', 'http', 'id', 'if', 'images', 'img', 'in', 'inc', 'increase', 'index', 'info', 'information', 'international', 'internet', 'into', 'investment', 'is', 'issue', 'issues', 'it', 'its', 'jackie', 'jan', 'january', 'jpg', 'julie', 'just', 'keep', 'know', 'last', 'let', 'life', 'like', 'limited', 'line', 'link', 'lisa', 'list', 'll', 'lloyd', 'long', 'look', 'looking', 'lose', 'loss', 'low', 'luong', 'made', 'mail', 'make', 'management', 'many', 'mar', 'march', 'market', 'mary', 'may', 'me', 'meds', 'meeting', 'melissa', 'message', 'meter', 'meters', 'methanol', 'meyers', 'mg', 'microsoft', 'midcon', 'million', 'mmbtu', 'monday', 'money', 'month', 'moopid', 'more', 'morris', 'most', 'ms', 'much', 'music', 'my', 'name', 'natural', 'nbsp', 'nd', 'need', 'needed', 'needs', 'net', 'new', 'news', 'next', 'no', 'nom', 'nomination', 'noms', 'north', 'not', 'note', 'now', 'number', 'of', 'off', 'offer', 'offers', 'office', 'on', 'once', 'one', 'online', 'only', 'operations', 'or', 'order', 'other', 'our', 'out', 'over', 'own', 'pain', 'paliourg', 'pat', 'path', 'pec', 'people', 'per', 'pg', 'photoshop', 'php', 'pills', 'pipeline', 'place', 'plant', 'please', 'pm', 'point', 'pops', 'prescription', 'price', 'prices', 'private', 'pro', 'problem', 'product', 'production', 'products', 'professional', 'purchase', 'put', 'quality', 'questions', 'ranch', 'rates', 're', 'receipt', 'receive', 'reliantenergy', 'remove', 'removed', 'reply', 'report', 'request', 'required', 'results', 'retail', 'right', 'risk', 'robert', 'sale', 'sales', 'same', 'save', 'scheduled', 'section', 'securities', 'security', 'see', 'send', 'sent', 'service', 'services', 'set', 'shares', 'she', 'shipping', 'should', 'show', 'since', 'sitara', 'site', 'size', 'smith', 'so', 'software', 'some', 'someone', 'soon', 'spam', 'special', 'src', 'statements', 'stella', 'still', 'stock', 'stocks', 'stop', 'strong', 'subject', 'such', 'suite', 'super', 'support', 'sure', 'susan', 'system', 'table', 'take', 'taylor', 'td', 'team', 'texas', 'th', 'than', 'thank', 'thanks', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'think', 'this', 'through', 'thu', 'ticket', 'tickets', 'time', 'to', 'today', 'tom', 'top', 'total', 'tr', 'transfer', 'transport', 'two', 'unify', 'united', 'until', 'up', 'us', 'use', 'valero', 'valium', 'vance', 've', 'very', 'via', 'viagra', 'visit', 'volume', 'volumes', 'want', 'was', 'we', 'web', 'week', 'weight', 'well', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'width', 'will', 'windows', 'with', 'within', 'without', 'work', 'works', 'world', 'worldwide', 'would', 'www', 'xanax', 'xls', 'xlssubject', 'xp', 'year', 'you', 'your']
resultFile = open("words.csv",'wb')
wr = csv.writer(resultFile, dialect='excel')
wr.writerow(features)
np.savetxt("features.csv",theta,delimiter=",")
'''



