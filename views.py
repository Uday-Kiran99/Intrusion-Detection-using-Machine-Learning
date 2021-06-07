from django.shortcuts import render,redirect

# Create your views here.
from projectuser.models import user_reg, ddos_dataset
import re
from django.db.models import Q, Count
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

def user_index(request):

    return render(request, 'projectuser/user_index.html')

def user_login(request):
    if request.method == "POST":
        uname = request.POST.get('uname')
        pswd = request.POST.get('password')
        try:
            check = user_reg.objects.get(uname=uname, password=pswd)
            request.session['uid'] = check.id
            request.session['uname'] = check.uname
            request.session['uemail'] = check.email
            return redirect('user_home')
        except:
            pass
        return redirect('user_login')
    return render(request, 'projectuser/user_login.html')

def user_register(request):
    if request.method == "POST":
        fullname = request.POST.get('fname')
        lname = request.POST.get('lname')
        email = request.POST.get('email')
        mobile = request.POST.get('mobile')
        uname = request.POST.get('uname')
        password = request.POST.get('password')
        user_reg.objects.create(fname=fullname, lname=lname,email=email, mobile=mobile, uname=uname,
                                password=password)
        return redirect('user_login')
    return render(request, 'projectuser/user_register.html')

def user_home(request):
    obj = ddos_dataset.objects.all()
    return render(request, 'projectuser/user_home.html',{'object':obj})


def add_data(request):
    attack1 = []
    attack2, attack3, attack4, attack5, attack6, attack7, attack8, attack9 = [], [], [], [], [], [], [], []
    ans = ''
    txt = ''
    splt = ''
    if request.method == "POST":
        txt = request.POST.get("name")

        splt = (re.findall(r"[\w']+", str(txt)))

    for f in splt:
        if f in ('IPid', 'FDDI', 'x25', 'rangingdistance'):
            attack1.append(f)
        elif f in ('tcpchecksum', 'mtcp', 'controlflags', 'tcpoffset', 'tcpport'):
            attack2.append(f)
        elif f in ('ICMPID', 'udptraffic', 'udpunicorn', 'datagramid', 'NTP', 'RIP', 'TFTP'):
            attack3.append(f)
        elif f in ('GETID', 'POSTID', 'openBSD', 'appid', 'sessionid', 'transid', 'physicalid'):
            attack4.append(f)
        elif f in ('SYN', 'ACK', 'synpacket', 'sycookies'):
            attack5.append(f)
        elif f in ('serverattack', 'serverid', 'blockbankwidth'):
            attack6.append(f)
        elif f in ('monlist', 'getmonlist', 'NTPserver'):
            attack7.append(f)
        elif f in ('portid', 'FTPID', 'tryion', 'fragflag'):
            attack8.append(f)
        elif f in ('malwareid', 'gethttpid', 'httpid'):
            attack9.append(f)

    if len(attack1) > len(attack2) and len(attack1) > len(attack3) and len(attack1) > len(attack4) and len(
            attack1) > len(attack5) and len(attack1) > len(attack6) and len(attack1) > len(attack7) and len(
        attack1) > len(attack8) and len(attack1) > len(attack9):
        ans = "Ip Fragment Attack"
    elif len(attack2) > len(attack1) and len(attack2) > len(attack3) and len(attack2) > len(attack4) and len(
            attack2) > len(attack5) and len(attack2) > len(attack6) and len(attack2) > len(attack7) and len(
        attack2) > len(attack8) and len(attack2) > len(attack9):
        ans = "TCP Based Attack"
    elif len(attack3) > len(attack2) and len(attack3) > len(attack1) and len(attack3) > len(attack4) and len(
            attack1) > len(attack5) and len(attack1) > len(attack6) and len(attack1) > len(attack7) and len(
        attack1) > len(attack8) and len(attack1) > len(attack9):
        ans = "UDP Based Attack"
    elif len(attack4) > len(attack2) and len(attack4) > len(attack3) and len(attack4) > len(attack1) and len(
            attack4) > len(attack5) and len(attack4) > len(attack6) and len(attack4) > len(attack7) and len(
        attack4) > len(attack8) and len(attack4) > len(attack9):
        ans = "Layer Based Attack"
    elif len(attack5) > len(attack2) and len(attack5) > len(attack3) and len(attack5) > len(attack4) and len(
            attack5) > len(attack1) and len(attack5) > len(attack6) and len(attack5) > len(attack7) and len(
        attack5) > len(attack8) and len(attack5) > len(attack9):
        ans = "SYN Floods Attack"
    elif len(attack6) > len(attack2) and len(attack6) > len(attack3) and len(attack6) > len(attack4) and len(
            attack6) > len(attack5) and len(attack6) > len(attack1) and len(attack6) > len(attack7) and len(
        attack6) > len(attack8) and len(attack6) > len(attack9):
        ans = "Slowloris"
    elif len(attack7) > len(attack2) and len(attack7) > len(attack3) and len(attack7) > len(attack4) and len(
            attack7) > len(attack5) and len(attack7) > len(attack6) and len(attack7) > len(attack1) and len(
        attack7) > len(attack8) and len(attack7) > len(attack9):
        ans = "NTP Amplification"
    elif len(attack8) > len(attack2) and len(attack8) > len(attack3) and len(attack8) > len(attack4) and len(
            attack8) > len(attack5) and len(attack8) > len(attack6) and len(attack8) > len(attack7) and len(
        attack8) > len(attack1) and len(attack8) > len(attack9):
        ans = "Ping of Death Attack"
    elif len(attack9) > len(attack2) and len(attack9) > len(attack3) and len(attack9) > len(attack4) and len(
            attack9) > len(attack5) and len(attack9) > len(attack6) and len(attack9) > len(attack7) and len(
        attack9) > len(attack8) and len(attack9) > len(attack1):
        ans = "HTTP Flood Attack"

    else:
        ans = "Unlabed Data"
    ddos_dataset.objects.create(ddos_data=txt, attack_result=ans)
    return render(request, 'projectuser/add_data.html')

def labeled_data(request):
    obj = ddos_dataset.objects.filter(Q(attack_result='Ip Fragment Attack') | Q(attack_result='TCP Based Attack') | Q(
        attack_result='UDP Based Attack') | Q(attack_result='NTP Amplification') | Q(
        attack_result='HTTP Flood Attack') | Q(attack_result='Layer Based Attack') | Q(attack_result='Slowloris') | Q(
        attack_result='Ping of Death Attack') | Q(attack_result='SYN Floods Attack'))
    return render(request, 'projectuser/labeled_data.html',{'object':obj})

def unlabeled_data(request):
    obj = ddos_dataset.objects.filter(attack_result='Unlabed Data')
    return render(request, 'projectuser/unlabeled_data.html',{'object':obj})

def intrusion_analysis(request):
    intrusion = ddos_dataset.objects.values('attack_result').annotate(dcount=Count('attack_result'))

    return render(request, 'projectuser/intrusion_analysis.html',{'object':intrusion})

def graphical_analysis(request):
    chart = ddos_dataset.objects.values('attack_result').annotate(dcount=Count('attack_result'))

    return render(request, 'projectuser/graphical_analysis.html',{'objects':chart})

def random_forest(request):
    df = pd.read_csv("network_dataset.csv")

    df.head()
    print(df.head())
    df.shape
    print(df.shape)
    df.isnull().sum()
    print(df.isnull().sum())
    num_cols = df._get_numeric_data().columns
    cate_cols = list(set(df.columns) - set(num_cols))

    cate_cols.remove('class')

    cate_cols

    print(cate_cols)

    df = df.dropna('columns')  # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values

    corr = df.corr()

    # This variable is highly correlated with num_compromised and should be ignored for analysis.
    # (Correlation = 0.9938277978738366)
    df.drop('num_root', axis=1, inplace=True)

    # This variable is highly correlated with serror_rate and should be ignored for analysis.
    # (Correlation = 0.9983615072725952)
    df.drop('srv_serror_rate', axis=1, inplace=True)

    # This variable is highly correlated with rerror_rate and should be ignored for analysis.
    # (Correlation = 0.9947309539817937)
    df.drop('srv_rerror_rate', axis=1, inplace=True)

    # This variable is highly correlated with srv_serror_rate and should be ignored for analysis.
    # (Correlation = 0.9993041091850098)
    df.drop('dst_host_srv_serror_rate', axis=1, inplace=True)

    # This variable is highly correlated with rerror_rate and should be ignored for analysis.
    # (Correlation = 0.9869947924956001)
    df.drop('dst_host_serror_rate', axis=1, inplace=True)

    # This variable is highly correlated with srv_rerror_rate and should be ignored for analysis.
    # (Correlation = 0.9821663427308375)
    df.drop('dst_host_rerror_rate', axis=1, inplace=True)

    # This variable is highly correlated with rerror_rate and should be ignored for analysis.
    # (Correlation = 0.9851995540751249)
    df.drop('dst_host_srv_rerror_rate', axis=1, inplace=True)

    # This variable is highly correlated with srv_rerror_rate and should be ignored for analysis.
    # (Correlation = 0.9865705438845669)
    df.drop('dst_host_same_srv_rate', axis=1, inplace=True)
    # protocol_type feature mapping
    pmap = {'icmp': 0, 'tcp': 1, 'udp': 2}
    df['protocol_type'] = df['protocol_type'].map(pmap)
    # flag feature mapping
    fmap = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTR': 3, 'RSTO': 4, 'SH': 5, 'S1': 6, 'S2': 7, 'RSTOS0': 8, 'S3': 9,
            'OTH': 10}
    df['flag'] = df['flag'].map(fmap)
    df.drop('service', axis=1, inplace=True)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    # Splitting the dataset

    # Target variable and train set
    y = df[['class']]
    X = df.drop(['class', ], axis=1)

    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    print(X)
    # Split test and train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    from sklearn.ensemble import RandomForestClassifier

    clfr = RandomForestClassifier(n_estimators=30)
    start_time = time.time()
    clfr.fit(X_train, y_train.values.ravel())
    end_time = time.time()
    print("Training time: ", end_time - start_time)
    start_time = time.time()
    y_test_pred = clfr.predict(X_train)
    end_time = time.time()
    print("Testing time: ", end_time - start_time)
    print("Train score is:", clfr.score(X_train, y_train))
    print("Test score is:", clfr.score(X_test, y_test))
    trainaccuracy=clfr.score(X_train, y_train)
    testaccuracy=clfr.score(X_test, y_test)
    plt.xlabel('Attack Prediction')
    plt.ylabel('Intrusion')
    plt.plot(clfr.predict(X_train), c='r')
    plt.title('Random Forest')
    plt.legend()
    plt.show()
    return render(request, 'projectuser/random_forest.html',{'trainaccuracy':trainaccuracy,'testaccuracy':testaccuracy})

def naive_bayes(request):
    df = pd.read_csv("network_dataset.csv")

    df.head()
    print(df.head())
    df.shape
    print(df.shape)
    df.isnull().sum()
    print(df.isnull().sum())
    num_cols = df._get_numeric_data().columns
    cate_cols = list(set(df.columns) - set(num_cols))

    cate_cols.remove('class')

    cate_cols

    print(cate_cols)

    df = df.dropna('columns')  # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values

    corr = df.corr()

    # This variable is highly correlated with num_compromised and should be ignored for analysis.
    # (Correlation = 0.9938277978738366)
    df.drop('num_root', axis=1, inplace=True)

    # This variable is highly correlated with serror_rate and should be ignored for analysis.
    # (Correlation = 0.9983615072725952)
    df.drop('srv_serror_rate', axis=1, inplace=True)

    # This variable is highly correlated with rerror_rate and should be ignored for analysis.
    # (Correlation = 0.9947309539817937)
    df.drop('srv_rerror_rate', axis=1, inplace=True)

    # This variable is highly correlated with srv_serror_rate and should be ignored for analysis.
    # (Correlation = 0.9993041091850098)
    df.drop('dst_host_srv_serror_rate', axis=1, inplace=True)

    # This variable is highly correlated with rerror_rate and should be ignored for analysis.
    # (Correlation = 0.9869947924956001)
    df.drop('dst_host_serror_rate', axis=1, inplace=True)

    # This variable is highly correlated with srv_rerror_rate and should be ignored for analysis.
    # (Correlation = 0.9821663427308375)
    df.drop('dst_host_rerror_rate', axis=1, inplace=True)

    # This variable is highly correlated with rerror_rate and should be ignored for analysis.
    # (Correlation = 0.9851995540751249)
    df.drop('dst_host_srv_rerror_rate', axis=1, inplace=True)

    # This variable is highly correlated with srv_rerror_rate and should be ignored for analysis.
    # (Correlation = 0.9865705438845669)
    df.drop('dst_host_same_srv_rate', axis=1, inplace=True)
    # protocol_type feature mapping
    pmap = {'icmp': 0, 'tcp': 1, 'udp': 2}
    df['protocol_type'] = df['protocol_type'].map(pmap)
    # flag feature mapping
    fmap = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTR': 3, 'RSTO': 4, 'SH': 5, 'S1': 6, 'S2': 7, 'RSTOS0': 8, 'S3': 9,
            'OTH': 10}
    df['flag'] = df['flag'].map(fmap)
    df.drop('service', axis=1, inplace=True)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    # Splitting the dataset

    # Target variable and train set
    y = df[['class']]
    X = df.drop(['class', ], axis=1)

    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    print(X)
    # Split test and train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    # Gaussian Naive Bayes
    clfg = GaussianNB()
    start_time = time.time()
    clfg.fit(X_train, y_train.values.ravel())
    end_time = time.time()
    print("Training time: ", end_time - start_time)

    start_time = time.time()
    y_test_pred = clfg.predict(X_test)
    print(y_test_pred)
    print(len(y_test_pred))

    end_time = time.time()
    print("Testing time: ", end_time - start_time)

    print("Train score is:", clfg.score(X_train, y_train))
    print("Test score is:", clfg.score(X_test, y_test))
    trainaccuracy = clfg.score(X_train, y_train)
    testaccuracy = clfg.score(X_test, y_test)
    plt.xlabel('Attack Prediction')
    plt.ylabel('Intrusion')
    plt.plot(clfg.predict(X_test), c='g')
    plt.title('Naive Bayes')
    plt.legend()
    plt.show()
    return render(request, 'projectuser/naive_bayes.html',{'trainaccuracy':trainaccuracy,'testaccuracy':testaccuracy})

def svm(request):
    df = pd.read_csv("network_dataset.csv")

    df.head()
    print(df.head())
    df.shape
    print(df.shape)
    df.isnull().sum()
    print(df.isnull().sum())
    num_cols = df._get_numeric_data().columns
    cate_cols = list(set(df.columns) - set(num_cols))

    cate_cols.remove('class')

    cate_cols

    print(cate_cols)

    df = df.dropna('columns')  # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values

    corr = df.corr()

    # This variable is highly correlated with num_compromised and should be ignored for analysis.
    # (Correlation = 0.9938277978738366)
    df.drop('num_root', axis=1, inplace=True)

    # This variable is highly correlated with serror_rate and should be ignored for analysis.
    # (Correlation = 0.9983615072725952)
    df.drop('srv_serror_rate', axis=1, inplace=True)

    # This variable is highly correlated with rerror_rate and should be ignored for analysis.
    # (Correlation = 0.9947309539817937)
    df.drop('srv_rerror_rate', axis=1, inplace=True)

    # This variable is highly correlated with srv_serror_rate and should be ignored for analysis.
    # (Correlation = 0.9993041091850098)
    df.drop('dst_host_srv_serror_rate', axis=1, inplace=True)

    # This variable is highly correlated with rerror_rate and should be ignored for analysis.
    # (Correlation = 0.9869947924956001)
    df.drop('dst_host_serror_rate', axis=1, inplace=True)

    # This variable is highly correlated with srv_rerror_rate and should be ignored for analysis.
    # (Correlation = 0.9821663427308375)
    df.drop('dst_host_rerror_rate', axis=1, inplace=True)

    # This variable is highly correlated with rerror_rate and should be ignored for analysis.
    # (Correlation = 0.9851995540751249)
    df.drop('dst_host_srv_rerror_rate', axis=1, inplace=True)

    # This variable is highly correlated with srv_rerror_rate and should be ignored for analysis.
    # (Correlation = 0.9865705438845669)
    df.drop('dst_host_same_srv_rate', axis=1, inplace=True)
    # protocol_type feature mapping
    pmap = {'icmp': 0, 'tcp': 1, 'udp': 2}
    df['protocol_type'] = df['protocol_type'].map(pmap)
    # flag feature mapping
    fmap = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTR': 3, 'RSTO': 4, 'SH': 5, 'S1': 6, 'S2': 7, 'RSTOS0': 8, 'S3': 9,
            'OTH': 10}
    df['flag'] = df['flag'].map(fmap)
    df.drop('service', axis=1, inplace=True)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    # Splitting the dataset

    # Target variable and train set
    y = df[['class']]
    X = df.drop(['class', ], axis=1)

    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    print(X)
    # Split test and train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    from sklearn.svm import SVC

    clfs = SVC(gamma='scale')
    start_time = time.time()
    clfs.fit(X_train, y_train.values.ravel())
    end_time = time.time()
    print("Training time: ", end_time - start_time)

    start_time = time.time()
    y_test_pred = clfs.predict(X_train)
    end_time = time.time()
    print("Testing time: ", end_time - start_time)

    print("Train score is:", clfs.score(X_train, y_train))
    print("Test score is:", clfs.score(X_test, y_test))
    trainaccuracy = clfs.score(X_train, y_train)
    testaccuracy = clfs.score(X_test, y_test)
    plt.xlabel('Attack Prediction')
    plt.ylabel('Intrusion')
    plt.plot(clfs.predict(X_test), c='b')
    plt.title('SVM')
    plt.legend()
    plt.show()
    return render(request, 'projectuser/svm.html',{'trainaccuracy':trainaccuracy,'testaccuracy':testaccuracy})