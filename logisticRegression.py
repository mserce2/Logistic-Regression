
"""
Bu projede bir kanserin iyi veya kötü huylu olarak sınıflandırılmış olan datasetini inceleyeceğiz
dataset üzerinde logistic regeression algoritmasını çalıştıracağız
"""

#%%Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%read csv
data = pd.read_csv("kanser_dataset.csv")

#Burda dataset üzerinde gereksiz yada işimize yaramayan column satırlardan kurtuluyoruz
data.drop(["Unnamed: 32","id"],axis=1,inplace = True)
#Burda ise kanserin iyi huylu olduğunu simgeleyen "m", kötü olduğunu
#simgeleyen "b" harflerini sayısal değere 0 ve 1 e dönüştürüyoruz
#çünkü işlem yaparken string olması algoritmanın çalışmasını engeller
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
print(data.info())

y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

#%% normalizasyon (x-min(x))/(max(x)-min(x))

"""
Normalizasyon; değerler arasında ki bozulmayı önlemek amacı ile
columnların değerlerini 0 ve 1 arasına çeker
"""
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

#%%train test split TrainTest.png resminde konu özeti vardır burda
#Datamızı uygun bir biçimde test etmemiz gerek
from sklearn.model_selection import train_test_split
#x ve y değerlerinin 0.2 yani %20 si test olarak al kalan %80 train olarak al
#yani x in %80'ini x_train,x in %20'si x_test
#aynı zaman da y classımızın labelları yani iyi huylu mu kötü huylumu ?
#x ise classımızın feture leri
#random_state=42 bu algoritmalar için kullanılan en genel değerdir
#eğer random_state kulanmassak datamız mantıklı bir şekilde %80 ve %20 lik kısımlarından
#bölünmez bu yüzden random_state=42 kullanıyoruz
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)

#Burada x_train x_test.... gibi veriable size (30,455) değil tam tersi bunu karmaşıklık
#önlenmesi amacı ile değiştiriyoruz. "size.png" resmine bakarsak klasor içindeki anlayabiliriz

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)

#%%parameter initialize and sigmoid function
#dimesion=30

def initialize_weights_and_bias(dimension):
    #burda dimesion [0.01] [0.01] ... kadar ekrana yazdıracağız
    #çünkü ağırlık 0 alırsak hata verir başlangıç olarak 0.01 alıyoruz
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b
#w,b=initialize_weights_and_bias(30)

# f(x) = 1/(1+(e^-x)) sigmoid fuction kodlayalım

def sigmoid(z):
    
    y_head = 1/(1+ np.exp(-z))
    return y_head
#print(sigmoid(0))  #0.5 cıkarsa hata yoktur demektir

#%%forward ve backward adımlarını kodlayalım

def forward_backward_propagation(w,b,x_train,y_train):
    #forward propagation z=b+px1w1+px2w2+...+pn+wn
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    #lost ve cost işlemlerini neden yaptığımızı lostcost.png resmine bakarak daha
    #iyi anlayabilirsiniz.Asıl amaç eğimi sıfıra yakın almak bu türev ile mevcut
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]  #x_train.shape[1] ölçekmele yapmak için kullandık
    
    #backward işlemleri: =>derivative:türev alma
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]  
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    return cost,gradients


    
#%%Updating(learning) parameters
#w,b,x,y_train değerlerinin daha önceden ne olduğunu öğrenmiştik
#burda bizim için yeni olan learning_rate=öğrenme kat sayısı ve number_of_iteration=ise
#kaç kez ileri geri(backward forward) yapcağımızı belirler
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    #parametleri şimdi number of iteration sayılarına göre update yapacağız
    for i in range(number_of_iterarion):
        #ileri ve geri giderken cost ve gradients değerlerini bulup update yapmamız gerek 
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        #tüm costları bir dizi içinde depoluyoruz
        cost_list.append(cost)
        
        #lest update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        
        #Şimdi her 10 adım da bir cost değerlerini saklayacağız
        #bunu modelin güvenirliği için yapıyoruz 10 tamamen random seçtik
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
        
        #Bu adımda bias ve ağırlıkları güncelleyeceğiz(w,b)
        parameters = {"weight": w,"bias": b}
        plt.plot(index,cost_list2)
        plt.xticks(index,rotation='vertical')
        plt.xlabel("Number of Iterarion")
        plt.ylabel("Cost")
        plt.show()
        return parameters, gradients, cost_list
    

#%%prediction
#Bu aşamada modelimizi test edeceğiz.Ne kadar verimli göreceğiz
def predict(w,b,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    #eğer z değerimiz 0.5 den küçük ise iyi huylu
    #değilse kötü huylu
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction


#%% logistic_regression
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    #initialize
    dimension =  x_train.shape[0]  # that is 30
    w,b = initialize_weights_and_bias(dimension)
    
    #learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    
    #Burda yaptığımız tesler % kaç oranında başarılı ekrana yazdırıyoruz
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 300)      
    
    
    
    
    
    

