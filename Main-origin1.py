import math
import heapq
import pickle
import random
import numpy as np   
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import PassiveAggressiveClassifier

    

sigma1=0.2
sigma2=0.2

#Read users information
with open('i://onedrive/Documents/project_dataset/pickle/x_train_alpha(0.005).pkl', 'rb') as f:
    user_data = pickle.load(f)

#Read Rotten
#ten = pd.read_csv('i://onedrive/Documents/project_dataset/archive/rotten_tomatoes_movies.csv')

#Read Rating
ratings = pd.read_csv('i://onedrive/Documents/project_dataset/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'rating_timestamp']
							).sort_values("rating_timestamp") # sorting the dataframe by datetime
ratings= ratings.drop('rating_timestamp',axis='columns')

#Read Genre
df_genre = pd.read_csv('i://onedrive/Documents/project_dataset/u.genre', sep='|', engine='python', names=['genre','ID'])
genre = df_genre['genre'].to_list()


a=['movie_id', 'movie_title', 'releasedate', 'videoreleasedate','URL']
for i in genre:
	a.append(i)

#Read movies information
movies = pd.read_csv('i://onedrive/Documents/project_dataset/u.item', sep='|',header=None, names= a ,encoding = "ISO-8859-1")

#Clustering movies
movies['year']=[0]*movies.shape[0]
for i in range(movies.shape[0]):
    date = str(movies.at[i,'releasedate'])
    if date == 'nan' :
        movies.at[i,'year']= '0'
    else:
        movies.at[i,'year']= date.split('-')[2]

data_origin = movies
data_origin = data_origin.drop(['movie_id','movie_title','releasedate','videoreleasedate','URL'],axis='columns')
for i in range(movies.shape[0]):
    if movies.at[i,'year']>='1990':
        #recently
        data_origin.at[i,'year']=1
    else:
        #not recently
        data_origin.at[i,'year']=0

data_origin = data_origin.to_numpy()
scaler = StandardScaler().fit(data_origin)
data_origin=scaler.transform(data_origin)
pca = PCA().fit(data_origin)
top_PCA=["%.2f" % a for a in pca.explained_variance_ratio_ if a >=0.01]
pca39 = PCA(n_components=len(top_PCA)).fit(data_origin)
Xpca=pca39.transform(data_origin)

#Clustering with kmeans as k=5
from sklearn.cluster import MiniBatchKMeans
km=MiniBatchKMeans(n_clusters=5,init='k-means++',max_iter=500,n_init=1000,init_size=1000,batch_size=1000,
                  verbose=False)
km_model=km.fit(Xpca)
kmeanlabels=km.labels_
kmeanclusters = km.predict(Xpca)
kmeandistances = km.transform(Xpca)

movies['movie_category']=[0]*movies.shape[0]
for i in range(movies.shape[0]):
    movies.at[i,'movie_category'] = kmeanclusters[i]+1

movies = movies.drop(['year'],axis='columns')

#Merge DataFrame
movies_rating = pd.merge(ratings,user_data, on='user_id')
total_movie_ratings = pd.merge(movies_rating , movies , on='movie_id')

print("Number of unique movie:")
print(total_movie_ratings['movie_title'].nunique())
total_movie_ratings = total_movie_ratings.drop(['movie_title','releasedate','videoreleasedate','URL'],axis='columns')


#model
Y = total_movie_ratings.iloc[:,2:3].to_numpy(dtype=int)
total = total_movie_ratings.drop(['user_id','movie_id','rating'], axis='columns')
X = total.to_numpy(dtype=float)

X ,X_test,Y,Y_test = train_test_split(X, Y, test_size=0.33,random_state=42)

model = RandomForestClassifier()
model.fit(X,Y.ravel())
#model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
#print(model.score(X,Y))

#from sklearn.metrics import classification_report
#Accuracy of model on Train dataset
#print(accuracy_score(y_pred,Y))
#print(classification_report(Y,y_pred))
#print(confusion_matrix(Y,y_pred))

y_pred = model.predict(X)

#print(classification_report(y_test,Y_test))
#print(confusion_matrix(Y_test,y_test))

confusion_matrix = metrics.confusion_matrix(Y, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [1,2,3,4,5])
#cm_display.plot()
#plt.show()


y_test = model.predict(X_test)

#Accuracy of model on Test dataset
confusion_matrix = metrics.confusion_matrix(Y_test,y_test)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [1,2,3,4,5])
#cm_display.plot()
#plt.show()

Y = np.reshape(Y,67000)

def root_mean_square(x,y):
    MSE = np.square(np.subtract(x,y)).mean()
    return math.sqrt(MSE)

print('RMSE for train dataset:'+str(root_mean_square(y_pred,Y)))
print('RMSE for test dataset:'+str(root_mean_square(y_test,Y_test)))
print('\n')

#save machine1
with open('i://onedrive/Documents/project_dataset/pickle/model1.pkl', 'wb') as f:
    pickle.dump(model,f)
#save machine2
with open('i://onedrive/Documents/project_dataset/pickle/model2.pkl', 'wb') as f:
    pickle.dump(model,f)
#save machine3
with open('i://onedrive/Documents/project_dataset/pickle/model3.pkl', 'wb') as f:
    pickle.dump(model,f)


#train_accuracy = accuracy_score(y_pred,Y)
#print("train acc:" + str(train_accuracy))
model_accuracy = accuracy_score(y_test,Y_test)
print("Accuracy on test dataset:"+ str(model_accuracy))


#Initial Expertise of user
#Find the number of each movie genre that the user has voted for
with open('i://onedrive/Documents/project_dataset/pickle/expertise1.pkl', 'rb') as f:
    expertise = pickle.load(f)

expertise['TotalTrans']=[0]*expertise.shape[0]
for i in range(expertise.shape[0]):
    sum=0
    for j in genre:
        sum += expertise.at[i,j]
    expertise.at[i,'TotalTrans']=sum

#Split Data into three dataset
total_movie_ratings['node'] = [0]*total_movie_ratings.shape[0]

dataset1_percent=.6
dataset2_percent=.2

nums = total_movie_ratings['user_id'].unique().tolist()
random.shuffle(nums)
    

data1 = nums[:round(dataset1_percent*len(nums))]
data2 = nums[round(dataset1_percent*len(nums)):round(dataset2_percent*len(nums)+dataset1_percent*len(nums))]
data3 = nums[round(dataset2_percent*len(nums)+dataset1_percent*len(nums)):]
    
for i in data1:
    total_movie_ratings.loc[total_movie_ratings['user_id']==i ,'node']= 1
for i in data2:
    total_movie_ratings.loc[total_movie_ratings['user_id']==i ,'node']= 2
for i in data3:
    total_movie_ratings.loc[total_movie_ratings['user_id']==i ,'node']= 3
    

dataset1 = total_movie_ratings.loc[total_movie_ratings['node']== 1]
dataset2 = total_movie_ratings.loc[total_movie_ratings['node']== 2]
dataset3 = total_movie_ratings.loc[total_movie_ratings['node']== 3]
nnum1 = len(set(dataset1['user_id'])&set(dataset2['user_id']))
nnum2 = len(set(dataset1['user_id'])&set(dataset3['user_id']))
nnum3 = len(set(dataset2['user_id'])&set(dataset3['user_id']))

print("Number of users shared between Node1 and Node2 : "+str(nnum1))
print("Number of users shared between Node1 and Node3 : "+str(nnum2))
print("Number of users shared between Node2 and Node3 : "+str(nnum3))
print('\n')


#Create rated_dataset
list_of_columns = list(total_movie_ratings.columns)
rated_dataset = pd.DataFrame(columns=list_of_columns)

#Initial user_trust
User_Trust = pd.DataFrame(columns=['user_id','trust','Num_Of_Correct_Trans','Total_Trans'])
for uid in user_data['user_id'].to_list():
    User_Trust.loc[len(User_Trust.index)] = [ uid , 0.2 , 0 , 0 ]

#Same product in nodes
dataset1 = total_movie_ratings.loc[total_movie_ratings['node']== 1]
dataset2 = total_movie_ratings.loc[total_movie_ratings['node']== 2]
dataset3 = total_movie_ratings.loc[total_movie_ratings['node']== 3]
num1 = len(set(dataset1['movie_id'])&set(dataset2['movie_id']))
num2 = len(set(dataset1['movie_id'])&set(dataset3['movie_id']))
num3 = len(set(dataset2['movie_id'])&set(dataset3['movie_id']))

print("Number of movies shared between Node1 and Node2 : "+str(num1))
print("Number of movies shared between Node1 and Node3 : "+str(num2))
print("Number of movies shared between Node2 and Node3 : "+str(num3))
print('\n')

#Initial Machine_Weight:
robot_weight = pd.DataFrame(columns=['node','movie_category','initial_weight','NumOfCorrectTrans','TotalTrans'])

for n in range(1,4):#node
    for c in range(1,6):#Cluster of movie
        #weight:number of movie with c category in node n/total number of movie in node n
        weight = (total_movie_ratings.loc[(total_movie_ratings['node']==n)&(total_movie_ratings['movie_category']==c),'movie_id'].unique().shape[0])/(total_movie_ratings.loc[total_movie_ratings['node']==n,'movie_id'].unique().shape[0])
        robot_weight.loc[len(robot_weight.index)] = [ n, c, model_accuracy*weight, 0, 0] 


#Initial number of transaction on each product  
NumberOfTransactionInNode = pd.DataFrame(columns=['node','movie_id','number'])
for movie in movies['movie_id'].unique().tolist():
    for index_of_node in range(1,4):
        #if not ((NumberOfTransactionInNode['node']==node) & (NumberOfTransactionInNode['movie_id']==movie)).any():
        NumberOfTransactionInNode.loc[len(NumberOfTransactionInNode.index)] = [index_of_node,movie,0]

#Create Local_score DataFrame
Local_score = pd.DataFrame(columns=['node','movie_id','local_score'])

#initial Global Score
    #اول کار که دیتای همه ربات ها یکی هست
    #global score = امتیاز ربات
    #predicted_score = model.predict(total_movie_ratings)#########اینجا اینکه چه کاربری بهش رای داده هم توش دخیله

Global_score = pd.DataFrame(columns=['movie_id','global_score'])
for movie in np.unique(total_movie_ratings['movie_id'].to_list()):
    initial_global_score = 2 #اینجا باید ویژگی های محصولو بدیم به مدل
    Global_score.loc[len(Global_score.index)] = [movie,initial_global_score] #initial global score = تجمیع امتیاز ربات نودها


#Shuffle the dataset
total_movie_ratings = total_movie_ratings.sample(frac = 1)
###اینجا باید حذف شه
#with open('i://onedrive/Documents/project_dataset/pickle/shuffle.pkl', 'rb') as f:
    #total_movie_ratings = pickle.load(f)

num_round = 1
while total_movie_ratings.shape[0]!=0:
    print(num_round)
    num_round += 1
    record_of_dataset = total_movie_ratings.iloc[0]
    total_movie_ratings = total_movie_ratings.drop([total_movie_ratings.index[0]],axis=0)
    rated_dataset.loc[len(rated_dataset)] = record_of_dataset.tolist()
    node = int(total_movie_ratings.iloc[0]['node'])
    user_id = int(record_of_dataset['user_id'])
    user_trust = User_Trust.loc[User_Trust['user_id']==user_id,'trust'].tolist()[0]
    user_rate = record_of_dataset['rating']
    rated_product = record_of_dataset['movie_id']
    movie_category = record_of_dataset['movie_category']
    
    movie_genre=[]
    for g in genre:
        if movies.loc[movies['movie_id']==rated_product,g].tolist()[0]==1:
            movie_genre.append(g)


    record_of_dataset = record_of_dataset.drop(['user_id','movie_id','node','rating']).to_numpy()
    

    #Generate Mahine's rate
    if node == 1:
        #load machine1
        with open('i://onedrive/Documents/project_dataset/pickle/model1.pkl', 'rb') as f:
            model = pickle.load(f)
    elif node == 2:
        #load machine2
        with open('i://onedrive/Documents/project_dataset/pickle/model2.pkl', 'rb') as f:
            model = pickle.load(f)
    elif node == 3:
        #load machine3
        with open('i://onedrive/Documents/project_dataset/pickle/model3.pkl', 'rb') as f:
            model = pickle.load(f)

    
    machine_rate = model.predict(record_of_dataset.reshape(1, -1))[0]
    if ((robot_weight['node']==node) & (robot_weight['movie_category']==movie_category) & (robot_weight['TotalTrans']==0)).any():
        machine_weight = robot_weight.loc[(robot_weight['node'] == node) & (robot_weight['movie_category']==movie_category) ]['initial_weight'].tolist()[0]
        
    else:
        machine_weight = robot_weight.loc[(robot_weight['node'] == node) & (robot_weight['movie_category']==movie_category)]['NumOfCorrectTrans'].tolist()[0]/robot_weight.loc[(robot_weight['node'] == node)&(robot_weight['movie_category']==movie_category)]['TotalTrans'].tolist()[0]
        
        
    #Compute LocalScore
    def Compute_Local_Score(user_id,movie_genre,user_rate,user_trust,machine_rate,machine_weight):
        if user_trust >= machine_weight :
            return user_rate
        elif user_trust < machine_weight:
            sum =0
            for g in movie_genre:
                sum += (expertise.loc[expertise['user_id']==user_id , g].tolist()[0]/expertise.loc[expertise['user_id']==user_id , 'TotalTrans'].tolist()[0])

            avg = sum/len(movie_genre)
            if avg >= 0.5:
                return user_rate
            return machine_rate

    local_score = Compute_Local_Score(user_id,movie_genre,int(user_rate),user_trust,int(machine_rate),machine_weight)
    if ((Local_score['node']==node) & (Local_score['movie_id']==rated_product)).any():
        Local_score.loc[(Local_score['node']== node) & (Local_score['movie_id']==rated_product),'local_score'] = local_score
    else:
        Local_score.loc[len(Local_score.index)] = [node,rated_product,local_score]

    #Retrain model with new score
    record_of_dataset = np.array([record_of_dataset])
    local_score = np.array([local_score])
    model.fit(record_of_dataset,local_score)

    #Save model
    if node == 1:
        #save machine1
        with open('i://onedrive/Documents/project_dataset/pickle/model1.pkl', 'wb') as f:
            pickle.dump(model,f)
    elif node == 2:
        #save machine2
        with open('i://onedrive/Documents/project_dataset/pickle/model2.pkl', 'wb') as f:
            pickle.dump(model,f)
    elif node == 3:
        #save machine3
        with open('i://onedrive/Documents/project_dataset/pickle/model3.pkl', 'wb') as f:
            pickle.dump(model,f)

    #Update Number of transaction in node
    NumberOfTransactionInNode.loc[(NumberOfTransactionInNode['node']==node) & ( NumberOfTransactionInNode['movie_id']==rated_product),'number'] +=1
    
    #Compute Similarity with other nodes
    similarity_vector = pd.DataFrame(columns=['node','Transaction_On_Product','E'])
    sglobal = Global_score.loc[Global_score['movie_id']==rated_product]['global_score'].tolist()[0]

    for index_of_node in range(1,4):
        if ((Local_score['node']== index_of_node) & (Local_score['movie_id']==rated_product)).any():
            slocal = Local_score.loc[(Local_score['node']==node) & (Local_score['movie_id'] == rated_product)]['local_score'].tolist()[0]
        else:
            if index_of_node==1:
                #load machine1
                with open('i://onedrive/Documents/project_dataset/pickle/model1.pkl', 'rb') as f:
                    model = pickle.load(f)
            elif index_of_node==2:
                #load machine2
                with open('i://onedrive/Documents/project_dataset/pickle/model2.pkl', 'rb') as f:
                    model = pickle.load(f)
            elif index_of_node==3:
                #load machine3
                with open('i://onedrive/Documents/project_dataset/pickle/model3.pkl', 'rb') as f:
                    model = pickle.load(f)

            slocal = model.predict(record_of_dataset.reshape(1, -1))[0]
        
        E = slocal / sglobal
        number = NumberOfTransactionInNode.loc[(NumberOfTransactionInNode['node']== index_of_node) & (NumberOfTransactionInNode['movie_id']==rated_product),'number'].tolist()[0]
        similarity_vector.loc[len(similarity_vector.index)] = [index_of_node,number,E]


    #Compute Cosine Similarity
    similarity=[-1]*len(similarity_vector)
    node_vector = similarity_vector.loc[similarity_vector['node']==node]
    node_vector = node_vector.drop(['node'],axis='columns').to_numpy().reshape(1,-1) 
    for index_of_node in range(1,4):
        if index_of_node!=node:
            
            similarity[index_of_node-1] = cosine_similarity(similarity_vector.loc[similarity_vector['node']== index_of_node].drop(['node'],axis='columns').to_numpy().reshape(1,-1),node_vector).tolist()[0][0]
        #elif index_of_node==node:
            #similarity[index_of_node]=-1

    #Find N most similar nodes to origin node
    similarity_backup=[]
    similarity_backup.extend(similarity)
    N = 2
    max_value = heapq.nlargest(N, similarity_backup)
    index_of_most_similar_node=[]
    for value in max_value:
        for index in range(0,len(similarity_backup)):
            if similarity_backup[index] == value:
                index_of_most_similar_node.append(index)
                similarity_backup[index]= -2
    index_of_most_similar_node = [ value+1 for value in index_of_most_similar_node]

    #Compute Score of nodes
    ScoreOfNode = pd.DataFrame(columns=['node','score'])
    for index_of_node in index_of_most_similar_node:
        if ((NumberOfTransactionInNode['node']==(index_of_node)) & (NumberOfTransactionInNode['movie_id']==rated_product) & (NumberOfTransactionInNode['number']!=0)).any(): #rated_product in np.unique(rated_dataset['movie_id'].to_list()):
            #Read local score of product from blockchain
            score = Local_score.loc[(Local_score['node']==index_of_node) & (Local_score['movie_id']==rated_product)]['local_score'].values[0]
        else:
            #Ask from robot of node
            if index_of_node==1:
                #load machine1
                with open('i://onedrive/Documents/project_dataset/pickle/model1.pkl', 'rb') as f:
                    model = pickle.load(f)
            elif index_of_node==2:
                #load machine2
                with open('i://onedrive/Documents/project_dataset/pickle/model2.pkl', 'rb') as f:
                    model = pickle.load(f)
            elif index_of_node==3:
                #load machine3
                with open('i://onedrive/Documents/project_dataset/pickle/model3.pkl', 'rb') as f:
                    model = pickle.load(f)
            score = model.predict(record_of_dataset.reshape(1, -1))[0]
        ScoreOfNode.loc[len(ScoreOfNode.index)] = [index_of_node,score]
    

    #Compute GlobalScore
    def Compute_Global_Score(ScoreOfNode ,similarity):
        sum_of_score=pd.DataFrame(columns=['rate','weight'])
        sum_of_score.loc[len(sum_of_score)]=[1,0]
        sum_of_score.loc[len(sum_of_score)]=[2,0]
        sum_of_score.loc[len(sum_of_score)]=[3,0]
        sum_of_score.loc[len(sum_of_score)]=[4,0]
        sum_of_score.loc[len(sum_of_score)]=[5,0]

        list_of_score = np.unique(ScoreOfNode['score']).tolist()
        for score in list_of_score:
            indexes = ScoreOfNode.loc[ScoreOfNode['score']==score,'node'].tolist()
            indexes = [int(index) for index in indexes]
            for i in indexes:
                sum_of_score.loc[sum_of_score['rate']==score,'weight'] += similarity[i-1]

        max_rate = sum_of_score.loc[sum_of_score['weight']==sum_of_score['weight'].max() ,'rate'].tolist()


        sum =0
        #If max weight belong to multiple score
        if len(max_rate)>1:
            #Average of value
            for i in max_rate:
                sum += i
            avg = round(sum/len(max_rate))
            return sum_of_score , avg
        else:
            return sum_of_score , max_rate[0]
    sum_of_score , Global_score.loc[Global_score['movie_id'] == rated_product,'global_score'] = Compute_Global_Score(ScoreOfNode,similarity)
    
    #Compute Trust of user
    def update_trust(sum_of_score,user_rate):
        weight = sum_of_score.loc[sum_of_score['rate']==user_rate]['weight'].tolist()[0]
        total_weight = sum_of_score['weight'].sum()
        return weight/total_weight

    correct_user_rate = False
    if Global_score.loc[Global_score['movie_id']==rated_product]['global_score'].tolist()[0] == user_rate:
        correct_user_rate = True
    #Update Trust of user
    User_Trust.loc[User_Trust['user_id']==user_id,'trust'] = update_trust(sum_of_score,user_rate)
    if correct_user_rate:
        User_Trust.loc[User_Trust['user_id']==user_id,'Num_Of_Correct_Trans']+= 1
    User_Trust.loc[User_Trust['user_id']==user_id,'Total_Trans'] += 1
          
    #Update Experience of user
    if correct_user_rate:
        for g in movie_genre:
            expertise.loc[expertise['user_id']==user_id , g] += 1
    expertise.loc[expertise['user_id']==user_id ,'TotalTrans'] += 1
    
    #Update weight of machine
    correct_machine_rate=False
    if Global_score.loc[Global_score['movie_id']==rated_product,'global_score'].tolist()[0] == machine_rate:
        correct_machine_rate=True
    if correct_machine_rate:
        robot_weight.loc[(robot_weight['node']== node) &(robot_weight['movie_category']==movie_category) ,'NumOfCorrectTrans'] += 1
    robot_weight.loc[(robot_weight['node']== node) &(robot_weight['movie_category']==movie_category) ,'TotalTrans'] += 1


    def Semi_Update_trust(User_Trust,user_id):
        #Compute trust
        numoftrans = User_Trust.loc[User_Trust['user_id']==user_id,'Num_Of_Correct_Trans'].tolist()[0]
        totaltrans = User_Trust.loc[User_Trust['user_id']==user_id,'Total_Trans'].tolist()[0]
        return numoftrans/totaltrans

    
    def Semi_Update_local_score(model , row_of_data):
        #هر کاربر به هر فیلم یکبار رای داده است
        #Compute robot feedback
        data = row_of_data.drop(['user_id','movie_id','node','rating'],axis='columns').to_numpy()
        machine_rate = model.predict(data.reshape(1, -1))[0]
        #Compute machine weight
        if ((robot_weight['node']==node) & (robot_weight['movie_category']==movie_category) & (robot_weight['TotalTrans']==0)).any():
            machine_weight = robot_weight.loc[(robot_weight['node'] == node) & (robot_weight['movie_category']==movie_category) ]['initial_weight'].tolist()[0]
        else:
            machine_weight = robot_weight.loc[(robot_weight['node'] == node) & (robot_weight['movie_category']==movie_category)]['NumOfCorrectTrans'].tolist()[0]/robot_weight.loc[(robot_weight['node'] == node)&(robot_weight['movie_category']==movie_category)]['TotalTrans'].tolist()[0]
    
        #Compute user feedback
        user_rate = row_of_data.iloc[0]['rating']
        user_id = row_of_data.iloc[0]['user_id']
        #Compute user trust
        user_trust = User_Trust.loc[User_Trust['user_id']==user_id,'trust'].tolist()[0]
        #Compute movie gere
        rated_product = row_of_data.iloc[0]['movie_id']
        movie_genre=[]
        for g in genre:
            if movies.loc[movies['movie_id']==rated_product,g].tolist()[0]==1:
                movie_genre.append(g)
        #Compute local score
        local_score = Compute_Local_Score(user_id,movie_genre,int(user_rate),user_trust,int(machine_rate),machine_weight)#index [0]
        return local_score
                

    def Semi_Update_global_score(node,record_of_dataset,product,Local_score,Global_score,NumberOfTransactionInNode):
        similarity_vector = pd.DataFrame(columns=['node','Transaction_On_Product','E'])

        sglobal = Global_score.loc[Global_score['movie_id']==rated_product]['global_score'].tolist()[0]

        for index_of_node in range(1,4):
            if ((Local_score['node']== index_of_node) & (Local_score['movie_id']==rated_product)).any():
                slocal = Local_score.loc[(Local_score['node']==node) & (Local_score['movie_id'] == rated_product)]['local_score'].tolist()[0]
            else:
                if index_of_node==1:
                    #load machine1
                    with open('i://onedrive/Documents/project_dataset/pickle/model1.pkl', 'rb') as f:
                        model = pickle.load(f)
                elif index_of_node==2:
                    #load machine2
                    with open('i://onedrive/Documents/project_dataset/pickle/model2.pkl', 'rb') as f:
                        model = pickle.load(f)
                elif index_of_node==3:
                    #load machine3
                    with open('i://onedrive/Documents/project_dataset/pickle/model3.pkl', 'rb') as f:
                        model = pickle.load(f)

                slocal = model.predict(record_of_dataset.reshape(1, -1))[0]
        
            E = slocal / sglobal
            number = NumberOfTransactionInNode.loc[(NumberOfTransactionInNode['node']== index_of_node) & (NumberOfTransactionInNode['movie_id']==rated_product),'number'].tolist()[0]
            similarity_vector.loc[len(similarity_vector.index)] = [index_of_node,number,E]


        #Compute Cosine Similarity
        similarity=[-1]*len(similarity_vector)
        node_vector = similarity_vector.loc[similarity_vector['node']==node]
        node_vector = node_vector.drop(['node'],axis='columns').to_numpy().reshape(1,-1) 
        for index_of_node in range(1,4):
            if index_of_node!=node:
            
                similarity[index_of_node-1] = cosine_similarity(similarity_vector.loc[similarity_vector['node']== index_of_node].drop(['node'],axis='columns').to_numpy().reshape(1,-1),node_vector).tolist()[0][0]
    
        #Find N most similar nodes to origin node
        similarity_backup=[]
        similarity_backup.extend(similarity)
        N = 2
        max_value = heapq.nlargest(N, similarity_backup)
        index_of_most_similar_node=[]
        for value in max_value:
            for index in range(0,len(similarity_backup)):
                if similarity_backup[index] == value:
                    index_of_most_similar_node.append(index)
                    similarity_backup[index]= -2
        index_of_most_similar_node = [ value+1 for value in index_of_most_similar_node]

        #Compute Score of nodes
        ScoreOfNode = pd.DataFrame(columns=['node','score'])
        for index_of_node in index_of_most_similar_node:
            if ((NumberOfTransactionInNode['node']==(index_of_node)) & (NumberOfTransactionInNode['movie_id']==rated_product) & (NumberOfTransactionInNode['number']!=0)).any(): #rated_product in np.unique(rated_dataset['movie_id'].to_list()):
                #Read local score of product from blockchain
                score = Local_score.loc[(Local_score['node']==index_of_node) & (Local_score['movie_id']==rated_product)]['local_score'].values[0]
            else:
                #Ask from robot of node
                if index_of_node==1:
                    #load machine1
                    with open('i://onedrive/Documents/project_dataset/pickle/model1.pkl', 'rb') as f:
                        model = pickle.load(f)
                elif index_of_node==2:
                    #load machine2
                    with open('i://onedrive/Documents/project_dataset/pickle/model1.pkl', 'rb') as f:
                        model = pickle.load(f)
                elif index_of_node==3:
                    #load machine3
                    with open('i://onedrive/Documents/project_dataset/pickle/model1.pkl', 'rb') as f:
                        model = pickle.load(f)
                score = model.predict(record_of_dataset.reshape(1, -1))[0]
            ScoreOfNode.loc[len(ScoreOfNode.index)] = [index_of_node,score]
        

        sum_of_score , global_score = Compute_Global_Score(ScoreOfNode,similarity)
        return global_score
    

    #Propagate score in the node with Semi Iterative Algorithm
    def SemiIterativeAlgorithm(node,rated_dataset,user_id,rated_product):
        RQ=pd.DataFrame(columns=['user_id','movie_id'])
        PQ=pd.DataFrame(columns=['movie_id','user_id'])
        
        Updated_Reviewer = []
        Updated_Product = []

        RQ.loc[len(RQ.index)] = [user_id,rated_product]
        while RQ.shape[0]!=0 or PQ.shape[0]!=0:
            while RQ.shape[0]!=0:

                ##باید trust قبلی رو نگه داریم
                old_trust = User_Trust.loc[User_Trust['user_id']==RQ.iloc[0]['user_id'],'trust'].tolist()[0]
                new_trust = Semi_Update_trust(User_Trust,RQ.iloc[0]['user_id'])
                User_Trust.loc[User_Trust['user_id']==RQ.iloc[0]['user_id'],'trust'] = new_trust
                
                #Get genre of product
                product_genre =[]
                for g in genre:
                    if movies.loc[movies['movie_id']==RQ.iloc[0]['movie_id'],g].tolist()[0]==1: 
                        product_genre.append(g)
                
                
                Updated_Reviewer.append(RQ.iloc[0]['user_id'])
                
                if abs(new_trust-old_trust)> sigma1:
                    if np.sign(new_trust-old_trust)>0:
                        User_Trust.loc[User_Trust['user_id']==RQ.iloc[0]['user_id'],'Num_Of_Correct_Trans'] += 1
                        #Update expertise
                        for g in product_genre:
                            expertise.loc[expertise['user_id']==RQ.iloc[0]['user_id'],g] += 1
                        
                    elif np.sign(new_trust-old_trust)<0:
                        User_Trust.loc[User_Trust['user_id']==RQ.iloc[0]['user_id'],'Num_Of_Correct_Trans'] -= 1
                        #Update expertise
                        for g in product_genre:
                            expertise.loc[expertise['user_id']==RQ.iloc[0]['user_id'],g] -= 1
                        
                    temp = rated_dataset.loc[(rated_dataset['node']==node) & (rated_dataset['user_id']==RQ.iloc[0]['user_id']) ,'movie_id'].tolist()
                    for movie in temp:
                        if movie not in Updated_Product:
                            PQ.loc[len(PQ.index)] = [movie,RQ.at[0,'user_id']]
                RQ = RQ.drop([RQ.index[0]],axis=0)
    
            while PQ.shape[0]!=0:

                if node == 1:
                    #load machine1
                    with open('i://onedrive/Documents/project_dataset/pickle/model1.pkl', 'rb') as f:
                        model = pickle.load(f)
                elif node == 2:
                    #load machine2
                    with open('i://onedrive/Documents/project_dataset/pickle/model2.pkl', 'rb') as f:
                        model = pickle.load(f)
                elif node == 3:
                    #load machine3
                    with open('i://onedrive/Documents/project_dataset/pickle/model3.pkl', 'rb') as f:
                        model = pickle.load(f)

                #Read old local score from ledger
                row_of_data = rated_dataset.loc[(rated_dataset['node']==node) & (rated_dataset['user_id']==PQ.iloc[0]['user_id']) & (rated_dataset['movie_id']==PQ.iloc[0]['movie_id'])]
                
                old_local_score = Local_score.loc[(Local_score['node']==node) & (Local_score['movie_id']==PQ.iloc[0]['movie_id']) ,'local_score'].tolist()[0]
                
                new_local_score = Semi_Update_local_score(model,row_of_data)
                
                Local_score.loc[(Local_score['node']==node) & (Local_score['movie_id']==PQ.iloc[0]['movie_id']),'local_score'] = new_local_score
                row_of_data = row_of_data.drop(['node','user_id','movie_id','rating'],axis='columns')
                row_of_data = row_of_data.to_numpy()
                
                #Retrain model with new local score
                model.fit(row_of_data,np.array([new_local_score]))

                if node == 1:
                    #save machine1
                    with open('i://onedrive/Documents/project_dataset/pickle/model1.pkl', 'wb') as f:
                        pickle.dump(model,f)
                elif node == 2:
                    #save machine2
                    with open('i://onedrive/Documents/project_dataset/pickle/model2.pkl', 'wb') as f:
                        pickle.dump(model,f)
                elif node == 3:
                    #save machine3
                    with open('i://onedrive/Documents/project_dataset/pickle/model3.pkl', 'wb') as f:
                        pickle.dump(model,f)

                
                Global_score.loc[Global_score['movie_id']==PQ.iloc[0]['movie_id'],'global_score'] = Semi_Update_global_score(node, row_of_data, PQ.iloc[0]['movie_id'], Local_score, Global_score, NumberOfTransactionInNode)

                Updated_Product.append(PQ.iloc[0]['movie_id'])
                if abs(new_local_score-old_local_score)> sigma2:
                    if np.sign(new_local_score-old_local_score)>0:
                        robot_weight.loc[(robot_weight['node']== node) &(robot_weight['movie_category']==movie_category) ,'NumOfCorrectTrans'] += 1
                    elif np.sign(new_local_score-old_local_score)<0:
                        
                        robot_weight.loc[(robot_weight['node']== node) &(robot_weight['movie_category']==movie_category) ,'NumOfCorrectTrans'] -= 1
                    
                    
                    temp = rated_dataset.loc[rated_dataset['movie_id']==PQ.iloc[0]['movie_id'],'user_id'].tolist()
                
                    for user in temp:
                        if user not in Updated_Reviewer:
                            RQ.loc[len(RQ.index)] = [user , PQ.at[0,'movie_id']]
                PQ = PQ.drop([PQ.index[0]],axis=0)
                
    
    SemiIterativeAlgorithm(node,rated_dataset,user_id,rated_product)

#Save local_score and user_trust in ledger or ...
            

#Propagate Score in Blockchain

#Save Global_score in a file
with open('i://onedrive/Documents/project_dataset/pickle/Global_score.pkl', 'wb') as f:
            pickle.dump(Global_score,f)
