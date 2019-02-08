import pandas as pd
import numpy as np
import pickle
import cv2
import os
import itertools
import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense,LSTM,Dropout,Bidirectional
from keras.layers import Input,Concatenate
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from  more_itertools import unique_everseen


trans=pd.read_csv("train.csv")

##Preprocessing to get ordered sequence of products purchased by each user##

#If Quantity of items purchased is n times repeating the same row n times
rep=trans[trans['Quantity']>1]

for i in range(0,len(rep)):
    quant=rep.iloc[i][2]
    j=0
    while j<quant-1:
        trans=trans.append(rep.iloc[i])
        j+=1

trans['OrderDate'] = pd.to_datetime(trans['OrderDate'])

#Group by on userId
grps=trans.groupby('UserId').groups

trans=np.array(trans)


dictList=[]

for key, value in grps.items():
    temp = value
    val=trans[temp,]
    val=val.tolist()
    dictList.append(val)

#Sorting based on time of purchase of item
for i in range(0,len(dictList)):
    dictList[i]=sorted(dictList[i],key=lambda x: x[3])



for i in range(0,len(dictList)):
    for j in range(0,len(dictList[i])):
        dictList[i][j]=dictList[i][j][1]
        dictList[i][j]=str(dictList[i][j])


##Representation for each product##


from gensim.models import Word2Vec

#Building word2vec on sequences of items purchased by each user
model_wv = Word2Vec(dictList,size =75,window = 4,min_count =1,iter=20)

##Representation for each user##
for_vectorizer=[]
for i in range(0,len(dictList)):
    for_vectorizer.append(" ".join(product for product in dictList[i]))


#Building countvectorizer for user-item matrix
vectorizer = CountVectorizer()
vec_user = vectorizer.fit_transform(for_vectorizer)

#Applying SVD on user-item matrix
svd = TruncatedSVD(n_components=256, n_iter=7, random_state=42)
svd.fit(vec_user)  


#Extracing features for users
svd_features=svd.transform(vec_user)


#Picking top 5 similar users for each UserId based on cosine similarity of user vectors
svd_temp = cosine_similarity(svd_features)
svd_top_ind = [np.argsort(i)[-4:-1] for i in svd_temp]


#Saving top 5 users for each user in pickle file
with open('svdd_top.pkl','wb') as f:
    pickle.dump(svd_top_ind, f)

##Extracting image features from VGG19##

#Reading images from folder and resizing them
def load_images_from_folder(folder):
    images = []
    images_names=[]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img = cv2.resize(img,(224,224))
            filename=filename.replace(".jpg","")
            images_names.append(filename)
            images.append(img)
    return images,images_names


images,image_names=load_images_from_folder("images/")

#Extracting features from pre-trained VGG-19 architecture
base_model = VGG19(weights='imagenet')
for i in range(0,len(images)):
    img=images[i]
    img=img.reshape((224,224,3))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    img=preprocess_input(img)
    img=img.reshape((224,224,3))
    images[i]=img

model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

images=np.array(images)

image_features = model.predict(images)

#Saving extracted image features into file.
with open('image_features.pkl','wb') as f:
    pickle.dump(image_features, f)


#image_features=np.load("image_features.pkl")
#svd_top_ind=np.load("svdd_top.pkl")


##Using product attributes##
pro_features=pd.read_csv("product_attributes.csv")

pro_df = pd.DataFrame(columns=pro_features['attribute_name'].unique(),index=pro_features['productid'].unique())

#Creating a attribute vector for each product
i = 0
while i < len(pro_features)-1:
    temp = pro_features['productid'][i]
    while temp == pro_features['productid'][i]:
        pro_df.loc[temp][pro_features['attribute_name'][i]] = pro_features['attributevalue'][i]
        i = i + 1
    continue
pro_df = pro_df.fillna(0)


#Training the recommendation system by taking recent purchase as target
target=[]
for i in range(0,len(dictList)):
    target.append(dictList[i][-1])


##Mixing all the features of the product to get a feature vector##

#Mixing all feature vectors to create a single representation for each item
feature_vectors=[]
for i in range(0,len(dictList)):
    ext_vec=[]
    #Considering recent 5 items purchased by user to predict the 6th item purchased
    for j in range(max(0,len(dictList[i])-6),len(dictList[i])-1):
        int_vec=[]
        #word2vec feature for item
        int_vec+=list(model_wv.wv[dictList[i][j]])
        
        #image feature vector
        try:
            ind=image_names.index(dictList[i][j])
            int_vec+=list(image_features[ind])
        except:
            int_vec+=[0]*4096
            
        #Product feature vector
        try:
            int_vec+=list(pro_df.loc[int(dictList[i][j])])
        except:
            int_vec+=[0]*9
            
        #Appending all the vectors representing a product into one
        ext_vec.append(int_vec)
        
    feature_vectors.append(ext_vec)



pad = [0]*4180

#Creating a fixed size input. i.e 5 length product sequences for each user.
feature_vectors = [[pad] * (5 - len(i)) + i for i in feature_vectors]


#One hot encoding of target
target_one_hot=pd.get_dummies(target)


##Model for recommending next item to purchase for the user##

#Model architecture
input1=Input(shape=(5,4180))
bi_lstm=Bidirectional(LSTM(128))(input1)
dense1=Dense(100,activation='relu')(bi_lstm)
input2=Input(shape=(9078,))
merged = Concatenate()([dense1,input2])
dense2=Dense(512,activation='relu')(merged)
output=Dense(len(set(target)),activation='softmax')(dense2)


full_model=Model(inputs=[input1,input2],outputs=output)

full_model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')


#Extracting top 3 similar users for each userID and appending it to the output vector of lstm
users_vec=[]
for i in range(0,len(svd_top_ind)):
    vec=[]
    for j in range(0,3):
        vec+=list(vec_user[svd_top_ind[i][j]].toarray().tolist()[0])
    users_vec.append(vec)


feature_vectors=np.array(feature_vectors)
users_vec=np.array(users_vec)

#Training the model
full_model.fit(batch_size=64,epochs=10,x=[feature_vectors,users_vec],y=target_one_hot)

all_cols=np.array(target_one_hot.columns)


##Recommendations for test data##

#Reading the test data
test_data=pd.read_csv("test_nFNPSyV.csv")

test_ind=list(test_data['UserId'])

#Extracting feature vectors for test data
features=feature_vectors[test_ind]

target=np.array(target)
target=target[test_ind]

#Adding the productID used as target in training as input to test data
#Extracting wrd2vec, image and product attributes
last_pro_feature_vector=[]
for i in range(0,len(target)):
    int_vec=[]
    int_vec+=list(model_wv.wv[target[i]])
    try:
        ind=image_names.index(target[i])
        int_vec+=list(image_features[ind])
    except:
        int_vec+=[0]*4096
    try:
        int_vec+=list(pro_df.loc[int(target[i])])
    except:
        int_vec+=[0]*9
        
    last_pro_feature_vector.append(int_vec)


for i in range(0,len(features)):
    for j in range(0,4):
        features[i][j]=features[i][j+1]
    features[i][4]=last_pro_feature_vector[i]


#User features also extracted
user_feat=users_vec[test_ind]

#predicting on test data
pred_test=full_model.predict([features,user_feat])

#Taking top 10 items predicted by softmax
pr_t=[]
for i in range(0,len(pred_test)):
    val=np.argsort(pred_test[i])[-10:]
    pr_t.append(list(all_cols[val]))


for i in range(0,len(pr_t)):
    for j in range(0,len(pr_t[i])):
        pr_t[i][j]=int(pr_t[i][j])
    pr_t[i]=pr_t[i][::-1]

#Saving the top 10 predictions into a file
final=pd.DataFrame({'UserId':test_ind,'product_list':pr_t})

final.to_csv("predict_1.csv",index=False)

##The above model gave much weightage to product features.

##Another model exclusively based on user similarity##

#Building a TFIDf vectorizer on user-item matrix
vectorizer=TfidfVectorizer()
vec_features = vectorizer.fit_transform(for_vectorizer)


#Taking cosine similarity on user vectors obtained
temp = cosine_similarity(vec_features)


#Checking commonly purchased items by top similar users
def func(count,similar,val):
    
    global dictList
    a = 11 * count
    b = a - 10
    sim = np.argsort(similar)[-a:-b]
    items = np.array(dictList)[sim]
    items = list(itertools.chain(*items))
    items = items[::-1]  
    items = collections.Counter(items).most_common()
    return [i[0] for i in items]

#Reading test data
tes_dat = pd.read_csv('test_nFNPSyV.csv')
test_ind=list(tes_dat['UserId'])

#Checking top 10 similar users and recommending products purchased by similar users
result = []
flag = 0
for i in list(tes_dat['UserId']):
    count = 0
    similar = temp[i]
    val = i
    bn = []
    while len(bn) <= 10:
        count = count + 1
        bn = bn + func(count,similar,val)
        bn = list(unique_everseen(bn))
    result.append(bn[0:10])
    flag = flag+1

for i in range(len(result)):
    for j in range(len(result[i])):
        result[i][j] = int(result[i][j])


#Writing the predictions to data frame and file
final2=pd.DataFrame({'UserId':list(tes_dat['UserId']),'product_list':result})
final2.to_csv("predict_2.csv",index=False)


##Ensembling recommendations from both the models##

po1=pd.read_csv("predict_2.csv")
po2=pd.read_csv("predict_1.csv")


#Reading the two prediction files.
pro1=po1['product_list']
pro2=po2['product_list']



new_pro1=[]
new_pro2=[]
for i in range(0,len(pro1)):
    p1=pro1[i].split(" ")
    p2=pro2[i].split(" ")
    
    p1[0]=p1[0].replace("[","")
    p2[0]=p2[0].replace("[","")
    
    p1[9]=p1[9].replace("]","")
    p2[9]=p2[9].replace("]","")
    
    for j in range(0,len(p1)):
        p1[j]=p1[j].replace(",","")
        p2[j]=p2[j].replace(",","")
        p1[j]=int(p1[j])
        p2[j]=int(p2[j])
    
    new_pro1.append(p1)
    new_pro2.append(p2)


#Picking the top 5 items from first model and top 5 items from second model as recommended products for user
new_pro=[]
for i in range(0,len(new_pro1)):
    pro=[]
    j=0
    k=0
    count=0
    while(count<20):
        if count%2==0:
            pro.append(new_pro1[i][j])
            j+=1
        else:
            pro.append(new_pro2[i][k])
            k+=1
            
        count+=1
            
    new_pro.append(pro)   


final_pro=[]
for i in range(0,len(new_pro)):
    final_pro.append(list(unique_everseen(new_pro[i]))[0:10])

#Writing final recommendations in a file
final=pd.DataFrame({'UserId':test_ind,'product_list':final_pro})
final.to_csv("final_prediction.csv",index=False)

