from tkinter import *
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np




main_win = Tk()

data=pd.read_csv('input.csv')


def Weight_Loss():
    print(" Age: %s\n Weight%s\n Height%s\n" % (e1.get(), e2.get(),e3.get()))
    USER_INP = simpledialog.askstring(title="Food Timing",
                        prompt="Enter 1 for Breakfast, 2 for Lunch and 3 for Dinner")
    
    Breakfastdata=data['Breakfast']
    BreakfastdataNumpy=Breakfastdata.to_numpy()
    
    Lunchdata=data['Lunch']
    LunchdataNumpy=Lunchdata.to_numpy()
    
    Dinnerdata=data['Dinner']
    DinnerdataNumpy=Dinnerdata.to_numpy()
    
    Food_itemsdata=data['Food_items']
    breakfastfoodseparated=[]
    Lunchfoodseparated=[]
    Dinnerfoodseparated=[]
    
    breakfastfoodseparatedID=[]
    LunchfoodseparatedID=[]
    DinnerfoodseparatedID=[]
    
    for i in range(len(Breakfastdata)):
      if BreakfastdataNumpy[i]==1: 
        breakfastfoodseparated.append(Food_itemsdata[i])
        breakfastfoodseparatedID.append(i)
      if LunchdataNumpy[i]==1:
        Lunchfoodseparated.append(Food_itemsdata[i])
        LunchfoodseparatedID.append(i)
      if DinnerdataNumpy[i]==1:
        Dinnerfoodseparated.append(Food_itemsdata[i])
        DinnerfoodseparatedID.append(i)
    
    
    
    # retrieving rows by loc method 
    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID]
    print(LunchfoodseparatedID)
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.iloc[Valapnd]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
    print (LunchfoodseparatedIDdata)
    print (LunchfoodseparatedIDdata.describe())
    
    # retrieving rows by loc method 
    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.iloc[Valapnd]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    print (breakfastfoodseparatedIDdata)
    print (breakfastfoodseparatedIDdata.describe())
    
    # retrieving rows by loc method 
    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.iloc[Valapnd]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
    print (DinnerfoodseparatedIDdata)
    print (DinnerfoodseparatedIDdata.describe())
    
    
    age=int(e1.get())
    weight=float(e2.get())
    height=float(e3.get())
    bmi = weight/(height**2) 
    
    for lp in range (0,80,20):
        test_list=np.arange(lp,lp+20)
        for i in test_list: 
            if(i == age):
                print('age is between',str(lp),str(lp+10))
                agecl=round(lp/20)    

    print("Your body mass index is: ", bmi)
    if ( bmi < 16):
        print("severely underweight")
        clbmi=4
    elif ( bmi >= 16 and bmi < 18.5):
        print("underweight")
        clbmi=3
    elif ( bmi >= 18.5 and bmi < 25):
        print("Healthy")
        clbmi=2
    elif ( bmi >= 25 and bmi < 30):
        print("overweight")
        clbmi=1
    elif ( bmi >=30):
        print("severely overweight")
        clbmi=0    
   
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.to_numpy()
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.to_numpy()
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.to_numpy()
    ti=(clbmi+agecl)/2
    
    ## K-Means Based Dinner Food
    Datacalorie=DinnerfoodseparatedIDdata[1:,1:len(DinnerfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    print ('* Prediction Result *')
    print(kmeans.labels_)
    print (kmeans.predict([Datacalorie[0]]))
    XValu=np.arange(0,len(kmeans.labels_))
    dnrlbl=kmeans.labels_
    
    ## K-Means Based Lunch Food
    Datacalorie=LunchfoodseparatedIDdata[1:,1:len(LunchfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    print ('* Prediction Result *')
    print(kmeans.labels_)
    XValu=np.arange(0,len(kmeans.labels_))
    lnchlbl=kmeans.labels_
    
    ## K-Means Based Breakfast Food
    Datacalorie=breakfastfoodseparatedIDdata[1:,1:len(breakfastfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    print ('## Prediction Result ##')
    print(kmeans.labels_)
    XValu=np.arange(0,len(kmeans.labels_))
    brklbl=kmeans.labels_
    
    
    datafin=pd.read_csv('inputfin.csv')
    
    ## train set
    #arrayfin=[agecl,clbmi,]
    #age bmi data combining and processed data(kmeans)
    dataTog=datafin.T
    bmicls=[0,1,2,3,4]
    agecls=[0,1,2,3,4]
    weightlosscat = dataTog.iloc[[1,2,7,8]]
    weightlosscat=weightlosscat.T
    weightgaincat= dataTog.iloc[[0,1,2,3,4,7,9,10]]
    weightgaincat=weightgaincat.T
    healthycat = dataTog.iloc[[1,2,3,4,6,7,9]]
    healthycat=healthycat.T
    weightlosscatDdata=weightlosscat.to_numpy()
    weightgaincatDdata=weightgaincat.to_numpy()
    healthycatDdata=healthycat.to_numpy()
    weightlosscat=weightlosscatDdata[1:,0:len(weightlosscatDdata)]
    weightgaincat=weightgaincatDdata[1:,0:len(weightgaincatDdata)]
    healthycat=healthycatDdata[1:,0:len(healthycatDdata)]
    
    weightlossfin=np.zeros((len(weightlosscat)*5,6),dtype=np.float32)
    weightgainfin=np.zeros((len(weightgaincat)*5,10),dtype=np.float32)
    healthycatfin=np.zeros((len(healthycat)*5,9),dtype=np.float32)
    t=0
    r=0
    s=0
    yt=[]
    yr=[]
    ys=[]
    for i in range(5):
        for j in range(len(weightlosscat)):
            valloc=list(weightlosscat[j])
            valloc.append(bmicls[i])
            valloc.append(agecls[i])
            weightlossfin[t]=np.array(valloc)
            yt.append(brklbl[j])
            t+=1
            weightlossfin[r]=np.array(valloc)
            yr.append(lnchlbl[j])
            r+=1
            weightlossfin[s]=np.array(valloc)
            ys.append(dnrlbl[j])
            s+=1
            
      
    X_test=np.zeros((len(weightlosscat),6),dtype=np.float32)

    print('**************')

    #Random Forest 
    for j in range(len(weightlosscat)):
        valloc=list(weightlosscat[j])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[j]=np.array(valloc)*ti
   
    val=int(USER_INP)
    
    if val==1:
        X_train= weightlossfin #Features
        y_train=yt #Labels
        
    elif val==2:
        X_train= weightlossfin
        y_train=yr 
        
    elif val==3:
        X_train= weightlossfin
        y_train=ys
        
    
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    
    
         
    print ('SUGGESTED FOOD ITEMS ::')
    for i in range(len(y_pred)):
        if y_pred[i]==2:
            print(Food_itemsdata[i])
    
   
def Weight_Gain():
    print(" Age: %s\n Weight%s\n Height%s\n" % (e1.get(), e2.get(),e3.get()))
    USER_INP = simpledialog.askstring(title="Food Timing",
                        prompt="Enter 1 for Breakfast, 2 for Lunch and 3 for Dinner")
    
    Breakfastdata=data['Breakfast']
    BreakfastdataNumpy=Breakfastdata.to_numpy()
    
    Lunchdata=data['Lunch']
    LunchdataNumpy=Lunchdata.to_numpy()
    
    Dinnerdata=data['Dinner']
    DinnerdataNumpy=Dinnerdata.to_numpy()
    
    Food_itemsdata=data['Food_items']
    breakfastfoodseparated=[]
    Lunchfoodseparated=[]
    Dinnerfoodseparated=[]
    
    breakfastfoodseparatedID=[]
    LunchfoodseparatedID=[]
    DinnerfoodseparatedID=[]
    
    for i in range(len(Breakfastdata)):
      if BreakfastdataNumpy[i]==1:
        breakfastfoodseparated.append(Food_itemsdata[i])
        breakfastfoodseparatedID.append(i)
      if LunchdataNumpy[i]==1:
        Lunchfoodseparated.append(Food_itemsdata[i])
        LunchfoodseparatedID.append(i)
      if DinnerdataNumpy[i]==1:
        Dinnerfoodseparated.append(Food_itemsdata[i])
        DinnerfoodseparatedID.append(i)
    
    # retrieving rows by loc method |
    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID]
    print(LunchfoodseparatedID)
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.iloc[Valapnd]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
    print (LunchfoodseparatedIDdata)
    print (LunchfoodseparatedIDdata.describe())
    
    # retrieving rows by loc method 
    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.iloc[Valapnd]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    print (breakfastfoodseparatedIDdata)
    print (breakfastfoodseparatedIDdata.describe())
    
    
    # retrieving rows by loc method 
    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.iloc[Valapnd]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
    print (DinnerfoodseparatedIDdata)
    print (DinnerfoodseparatedIDdata.describe())
   
    age=int(e1.get())
    weight=float(e2.get())
    height=float(e3.get())
    bmi = weight/(height**2) 
    
    for lp in range (0,80,20):
        test_list=np.arange(lp,lp+20)
        for i in test_list: 
            if(i == age):
                print('age is between',str(lp),str(lp+10))
                agecl=round(lp/20)    

    
    print("Your body mass index is: ", bmi)
    if ( bmi < 16):
        print("severely underweight")
        clbmi=4
    elif ( bmi >= 16 and bmi < 18.5):
        print("underweight")
        clbmi=3
    elif ( bmi >= 18.5 and bmi < 25):
        print("Healthy")
        clbmi=2
    elif ( bmi >= 25 and bmi < 30):
        print("overweight")
        clbmi=1
    elif ( bmi >=30):
        print("severely overweight")
        clbmi=0    
    val1=DinnerfoodseparatedIDdata.describe()
    valTog=val1.T
    print (valTog.shape)
    print (valTog)
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.to_numpy()
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.to_numpy()
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.to_numpy()
    ti=(bmi+agecl)/2
    
    
    ## K-Means Based  Dinner Food
    Datacalorie=DinnerfoodseparatedIDdata[1:,1:len(DinnerfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    print ('## Prediction Result ##')
    print(kmeans.labels_)
    print (kmeans.predict([Datacalorie[0]]))
    XValu=np.arange(0,len(kmeans.labels_))
    dnrlbl=kmeans.labels_
   
    ## K-Means Based Lunch Food
    Datacalorie=LunchfoodseparatedIDdata[1:,1:len(LunchfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    print ('## Prediction Result ##')
    print(kmeans.labels_)
    XValu=np.arange(0,len(kmeans.labels_))
    lnchlbl=kmeans.labels_

    ## K-Means Based Breakfast Food
    Datacalorie=breakfastfoodseparatedIDdata[1:,1:len(breakfastfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    print ('## Prediction Result ##')
    print(kmeans.labels_)
    XValu=np.arange(0,len(kmeans.labels_))
    brklbl=kmeans.labels_
    
    
    datafin=pd.read_csv('inputfin.csv')
    dataTog=datafin.T
    bmicls=[0,1,2,3,4]
    agecls=[0,1,2,3,4]
    weightlosscat = dataTog.iloc[[1,2,7,8]]
    weightlosscat=weightlosscat.T
    weightgaincat= dataTog.iloc[[0,1,2,3,4,7,9,10]]
    weightgaincat=weightgaincat.T
    healthycat = dataTog.iloc[[1,2,3,4,6,7,9]]
    healthycat=healthycat.T
    weightlosscatDdata=weightlosscat.to_numpy()
    weightgaincatDdata=weightgaincat.to_numpy()
    healthycatDdata=healthycat.to_numpy()
    weightlosscat=weightlosscatDdata[1:,0:len(weightlosscatDdata)]
    weightgaincat=weightgaincatDdata[1:,0:len(weightgaincatDdata)]
    healthycat=healthycatDdata[1:,0:len(healthycatDdata)]
    
   
    weightlossfin=np.zeros((len(weightlosscat)*5,6),dtype=np.float32)
    weightgainfin=np.zeros((len(weightgaincat)*5,10),dtype=np.float32)
    healthycatfin=np.zeros((len(healthycat)*5,9),dtype=np.float32)
    t=0
    r=0
    s=0
    yt=[]
    yr=[]
    ys=[]
    for i in range(5):
        for j in range(len(weightgaincat)):
            valloc=list(weightgaincat[j])
            valloc.append(bmicls[i])
            valloc.append(agecls[i])
            weightgainfin[t]=np.array(valloc)
            yt.append(brklbl[j])
            t+=1
            weightgainfin[r]=np.array(valloc)
            yr.append(lnchlbl[j])
            r+=1
            weightgainfin[s]=np.array(valloc)
            ys.append(dnrlbl[j])
            s+=1

    
    X_test=np.zeros((len(weightgaincat),10),dtype=np.float32)

    print('**************')
    #Random Forest
    for j in range(len(weightgaincat)):
        valloc=list(weightgaincat[j])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[j]=np.array(valloc)*ti
   
    
    val=int(USER_INP)
    
    if val==1:
        X_train= weightgainfin
        y_train=yt
        
    elif val==2:
        X_train= weightgainfin
        y_train=yr 
        
    elif val==3:
        X_train= weightgainfin
        y_train=ys
    
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    
        
    print ('SUGGESTED FOOD ITEMS ::')
    for i in range(len(y_pred)):
        if y_pred[i]==2:
            print (Food_itemsdata[i])
   
    
     
def Healthy():
    print(" Age: %s\n Weight%s\n Height%s\n" % (e1.get(), e2.get(),e3.get()))
    USER_INP = simpledialog.askstring(title="Food Timing",
                        prompt="Enter 1 for Breakfast, 2 for Lunch and 3 for Dinner")
    
    Breakfastdata=data['Breakfast']
    BreakfastdataNumpy=Breakfastdata.to_numpy()
    
    Lunchdata=data['Lunch']
    LunchdataNumpy=Lunchdata.to_numpy()
    
    Dinnerdata=data['Dinner']
    DinnerdataNumpy=Dinnerdata.to_numpy()
    
    Food_itemsdata=data['Food_items']
    breakfastfoodseparated=[]
    Lunchfoodseparated=[]
    Dinnerfoodseparated=[]
    
    breakfastfoodseparatedID=[]
    LunchfoodseparatedID=[]
    DinnerfoodseparatedID=[]
    
    for i in range(len(Breakfastdata)):
      if BreakfastdataNumpy[i]==1:
        breakfastfoodseparated.append(Food_itemsdata[i])
        breakfastfoodseparatedID.append(i)
      if LunchdataNumpy[i]==1:
        Lunchfoodseparated.append(Food_itemsdata[i])
        LunchfoodseparatedID.append(i)
      if DinnerdataNumpy[i]==1:
        Dinnerfoodseparated.append(Food_itemsdata[i])
        DinnerfoodseparatedID.append(i)
    
    
    # retrieving rows by loc method |
    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID]
    print(LunchfoodseparatedID)
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.iloc[Valapnd]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
    print (LunchfoodseparatedIDdata)
    print (LunchfoodseparatedIDdata.describe())
    
    # retrieving rows by loc method 
    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.iloc[Valapnd]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    print (breakfastfoodseparatedIDdata)
    print (breakfastfoodseparatedIDdata.describe())
    
    
    # retrieving rows by loc method 
    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.iloc[Valapnd]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
    print (DinnerfoodseparatedIDdata)
    print (DinnerfoodseparatedIDdata.describe())
    
    age=int(e1.get())
    weight=float(e2.get())
    height=float(e3.get())
    bmi = weight/(height**2) 
    
    for lp in range (0,80,20):
        test_list=np.arange(lp,lp+20)
        for i in test_list: 
            if(i == age):
                print('age is between',str(lp),str(lp+10))
                agecl=round(lp/20)    
    

    print("Your body mass index is: ", bmi)
    if ( bmi < 16):
        print("severely underweight")
        clbmi=4
    elif ( bmi >= 16 and bmi < 18.5):
        print("underweight")
        clbmi=3
    elif ( bmi >= 18.5 and bmi < 25):
        print("Healthy")
        clbmi=2
    elif ( bmi >= 25 and bmi < 30):
        print("overweight")
        clbmi=1
    elif ( bmi >=30):
        print("severely overweight")
        clbmi=0    
    
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.to_numpy()
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.to_numpy()
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.to_numpy()
    ti=(bmi+agecl)/2
    
    
    ## K-Means Based  Dinner Food
    Datacalorie=DinnerfoodseparatedIDdata[1:,1:len(DinnerfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    print ('## Prediction Result ##')
    print(kmeans.labels_)
    print (kmeans.predict([Datacalorie[0]]))
    XValu=np.arange(0,len(kmeans.labels_))
    dnrlbl=kmeans.labels_
   
    ## K-Means Based Lunch Food
    Datacalorie=LunchfoodseparatedIDdata[1:,1:len(LunchfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    print ('## Prediction Result ##')
    print(kmeans.labels_)
    XValu=np.arange(0,len(kmeans.labels_))
    lnchlbl=kmeans.labels_
    
    ## K-Means Based Breakfast Food
    Datacalorie=breakfastfoodseparatedIDdata[1:,1:len(breakfastfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    print ('## Prediction Result ##')
    print(kmeans.labels_)
    XValu=np.arange(0,len(kmeans.labels_))
    brklbl=kmeans.labels_
    
    
    
    datafin=pd.read_csv('G:\Major project\inputfin.csv')
    dataTog=datafin.T
    bmicls=[0,1,2,3,4]
    agecls=[0,1,2,3,4]
    weightlosscat = dataTog.iloc[[1,2,7,8]]
    weightlosscat=weightlosscat.T
    weightgaincat= dataTog.iloc[[0,1,2,3,4,7,9,10]]
    weightgaincat=weightgaincat.T
    healthycat = dataTog.iloc[[1,2,3,4,6,7,9]]
    healthycat=healthycat.T
    weightlosscatDdata=weightlosscat.to_numpy()
    weightgaincatDdata=weightgaincat.to_numpy()
    healthycatDdata=healthycat.to_numpy()
    weightlosscat=weightlosscatDdata[1:,0:len(weightlosscatDdata)]
    weightgaincat=weightgaincatDdata[1:,0:len(weightgaincatDdata)]
    healthycat=healthycatDdata[1:,0:len(healthycatDdata)]
    
    
    weightlossfin=np.zeros((len(weightlosscat)*5,6),dtype=np.float32)
    weightgainfin=np.zeros((len(weightgaincat)*5,10),dtype=np.float32)
    healthycatfin=np.zeros((len(healthycat)*5,9),dtype=np.float32)
    t=0
    r=0
    s=0
    yt=[]
    yr=[]
    ys=[]
    for i in range(5):
        for j in range(len(healthycat)):
            valloc=list(healthycat[j])
            valloc.append(bmicls[i])
            valloc.append(agecls[i])
            healthycatfin[t]=np.array(valloc)
            yt.append(brklbl[j])
            t+=1
            healthycatfin[r]=np.array(valloc)
            yr.append(lnchlbl[j])
            r+=1
            healthycatfin[s]=np.array(valloc)
            ys.append(dnrlbl[j])
            s+=1

    X_test=np.zeros((len(healthycat)*5,9),dtype=np.float32)
    print('**************')
   
    for j in range(len(healthycat)):
        valloc=list(healthycat[j])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[j]=np.array(valloc)*ti
    
    val=int(USER_INP)
    
    if val==1:
        X_train= healthycatfin
        y_train=yt
        
    elif val==2:
        X_train= healthycatfin
        y_train=yt 
        
    elif val==3:
        X_train= healthycatfin
        y_train=ys
        
    
   
    
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
   
    
    print ('SUGGESTED FOOD ITEMS ::')
    for ii in range(len(y_pred)):
        if y_pred[ii]==2:
            print (Food_itemsdata[ii])





#GUIPART
Label(main_win,text="Age",font='Helvetica 12 bold').grid(row=1,column=0,sticky=W,pady=4)
Label(main_win,text="Weight",font='Helvetica 12 bold').grid(row=2,column=0,sticky=W,pady=4)
Label(main_win,text="Height", font='Helvetica 12 bold').grid(row=3,column=0,sticky=W,pady=4)

e1 = Entry(main_win,bg="light grey")
e2 = Entry(main_win,bg="light grey")
e3 = Entry(main_win,bg="light grey")
e1.focus_force() 

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)


Button(main_win,text='Weight Loss',font='Helvetica 8 bold',command=Weight_Loss).grid(row=5,column=0,sticky="",pady=4)
Button(main_win,text='Weight Gain',font='Helvetica 8 bold',command=Weight_Gain).grid(row=5,column=1,sticky="",pady=4)
Button(main_win,text='Healthy',font='Helvetica 8 bold',command=Healthy).grid(row=5,column=2,sticky="",pady=4)
main_win.geometry("400x200")
main_win.wm_title("DIET RECOMMENDATION SYSTEM")
main_win.mainloop()