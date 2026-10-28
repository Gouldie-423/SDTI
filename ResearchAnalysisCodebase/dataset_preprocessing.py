import torch 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle
from sklearn.datasets import fetch_kddcup99
from PIL import Image
from scipy.io import loadmat

airline_path = 'datasets/full_data_flightdelay.csv'
korean_health_path = 'datasets/smoking_driking_dataset_Ver01.csv'
cifar10_path = "./datasets/cifar-10-batches-py"
fashion_mnist_path = "./datasets/fashion-mnist_train.csv"
digit_mnist_path = "./datasets/digit_mnist/"
blood_type_path = "./datasets/blood_type/"
covertype_path = "./datasets/covtype.data.gz"
shvn_path = "./datasets/shvn_train_32x32.mat"
crop_mapping_path = "./datasets/crop_mapping.txt"

def binary_classification_conversion(data,labels):
	export_data = []

	if len(np.unique(labels)) % 2 != 0:
		data = np.squeeze(data[np.where(labels != np.max(np.unique(labels))),:])
		labels = labels[np.where(labels != np.max(np.unique(labels)))]

	#converting data into n groups of binary classification tasks
	##labels are automatically converted to binary (0,1) values as well

	for i in range(1,len(np.unique(labels)),2):
		filtered_data = data[np.where((labels==i-1)|(labels==i))][:3000]
		filtered_labels = np.where(labels[np.where((labels==i-1)|(labels==i))] > np.min(labels[np.where((labels==i-1)|(labels==i))]), 1, 0)[:3000]
		export_data.append([filtered_data.astype(np.float32),filtered_labels.astype(np.int8)])
	return export_data
def airline_data_preprocessing(path):
	#loading and taking sample of data
	df = pd.read_csv(path)
	df = df.sample(n=3000,random_state=42)

	# Removing un-necessary columns
	## One-hot-encoding these columns would lead to 1000's of features, not ideal for meta learning experiment
	### Target variable (Is plane delayed more than 15 minutes) will be determined by weather conditions 
	del df['DEP_TIME_BLK']
	del df['CARRIER_NAME']
	del df['DEPARTING_AIRPORT']
	del df['PREVIOUS_AIRPORT']

	#seperating labels from rest of dataframe
	labels = df['DEP_DEL15']
	del df['DEP_DEL15']
	
	return df.to_numpy(),labels.to_numpy()

def korean_health_preprocessing(path):
	#loading and taking sample of data
	df = pd.read_csv(path)
	df = df.sample(n=3000,random_state=42)

	#one-hot-encoding required columns
	df = pd.get_dummies(df,columns=['sex','DRK_YN'],drop_first=True)

	#seperating labels from rest of dataframe
	labels = df['DRK_YN_Y']
	del df['DRK_YN_Y']
	
	return df.to_numpy(),labels.to_numpy()

def cifar10_preprocessing(path):

	def open_cifar(path,file): #function to open files corresponding to each batch
		with open(path+'/'+file, 'rb') as fo:
			df = pickle.load(fo, encoding='bytes')
			keys = list(df.keys())
			data = df[keys[2]]
			labels = np.array(df[keys[1]])
			
			return (data,labels)
	#only taking 5k records to limit run-time of algorithm during beta testing
	#used to shuffle for final performance tests
	sample = np.random.permutation(60000)

	#opening, stacking, reshaping, and shuffling a sample of both dataset and labels
	data = np.stack([open_cifar(path,file)[0] for file in os.listdir(path) if file != ".DS_Store"]).reshape(60000,3072)[sample,:]
	labels = np.stack([open_cifar(path,file)[1] for file in os.listdir(path) if file != ".DS_Store"]).reshape(60000)[sample]
	
	#converting multiclass data into a series of binary classification tasks
	##exports nested list object with each entry being [data,labels]
	export_data_object = binary_classification_conversion(data,labels)
	

	return export_data_object

def fashion_mnist_preprocessing(path):
	#loading and taking sample of data
	df = pd.read_csv(path)
	df = df.sample(n=60000,random_state=42)

	#seperating labels from rest of dataframe
	labels = df['label']
	del df['label']

	#converting multiclass data into a series of binary classification tasks
	##exports nested list object with each entry being [data,labels] 
	export_data_object = binary_classification_conversion(df.to_numpy(),labels.to_numpy())

	return export_data_object

def digit_mnist_preprocessing(path):
	data = pd.read_csv(path+'MNIST_data.csv')
	labels = pd.read_csv(path+'MNIST_target.csv')
	
	data = data.sample(n=len(data),random_state=42)
	labels = labels.sample(n=len(labels),random_state=42)

	export_data_object = binary_classification_conversion(data.to_numpy(),np.squeeze(labels.to_numpy()))

	return export_data_object

def blood_type_preprocessing(path):

	#need to unroll before putting into a dataframe
	images = []
	for folder in os.listdir(path):
		if folder != ".DS_Store":
			for file in os.listdir(path+folder):
				if file != '.DS_Store':
					image_path = os.path.join(path+folder+'/', file)
					image = Image.open(image_path)
					images.append(image)
					image.close()

	image_series = [pd.Series(image) for image in images]
	df = pd.DataFrame(image_series)
	print(df)

def covertype_preprocessing(path):

	data = pd.read_csv(path, compression='gzip')
	labels = data['5'] - 1 
	del data['5']
	
	data = data.sample(n=len(data),random_state=42)
	labels = labels.sample(n=len(labels),random_state=42)

	export_data_object = binary_classification_conversion(data.to_numpy(),np.squeeze(labels.to_numpy()))
	return export_data_object

def shvn_preprocessing(path):
	# Load .mat file
	file = loadmat(path)
	data = file['X'] #shape = (32, 32, 3, 73257)
	labels = file['y']

	#unrolling image data 
	data = np.reshape(data,(3072,data.shape[3]))
	#transposing data. Needs to be (records, features)
	data = data.T

	export_data_object = binary_classification_conversion(data,np.squeeze(labels))
	return export_data_object

def crop_mapping_preprocessing(path):
	data  = pd.read_csv(path,sep=',')
	labels = data['label']
	del data['label']

	export_data_object = binary_classification_conversion(data.to_numpy(),np.squeeze(labels.to_numpy()))
	return export_data_object

def normalize_and_create_metadata(data_dict):
    
    scaler = StandardScaler()
    train_datasets = {}
    
    keys = [i for i in data_dict.keys()]
    #filling in metadata
    for dataset in range(0,len(data_dict.keys())):
        #normalizing correct indicies based on fold of data
        train_data = scaler.fit_transform(data_dict[keys[dataset]][0][:2000]).astype(np.float32)
        
        #each dict entry is shaped [data,labels]
        train_datasets[keys[dataset]] = [train_data,data_dict[keys[dataset]][1][:2000]]        
        
    #generating labels for metadata based on index position of original dataset
    
    ground_truth_labels = np.stack([i for i in range(0,len(data_dict.keys()))])
    return train_datasets, ground_truth_labels

def zero_pad_datasets(data_dict):
	keys = [i for i in data_dict.keys()]
	max_size = max([data_dict[key][0].shape[1] for key in keys])
	for key in keys:
		if data_dict[key][0].shape[1] != max_size:
			missing_features_size = max_size - data_dict[key][0].shape[1]
			data_dict[key][0]  = np.concatenate((data_dict[key][0],np.zeros((data_dict[key][0].shape[0],missing_features_size),dtype=np.int8)),axis=1)

	return data_dict 

def data_structure_gen(data_dict):
	keys = [i for i in data_dict.keys()]
	data = np.stack([data_dict[keys[data]][0] for data in range(0,len(keys)) for label in range(0,len(keys))]).astype(np.float32)
	labels = np.stack([data_dict[keys[label]][1] for data in range(0,len(keys)) for label in range(0,len(keys))]).astype(np.int8)

	return data,labels

def load_datasets():
	#loading binary classification converted data
	cifar_data = cifar10_preprocessing(cifar10_path)
	fashion_mnist_data = fashion_mnist_preprocessing(fashion_mnist_path)
	digit_mnist_data = digit_mnist_preprocessing(digit_mnist_path)
	#all datasets are shaped (records,features)
	data_dict = {'Korean Health':korean_health_preprocessing(korean_health_path),
	'Airline Delay Data':airline_data_preprocessing(airline_path),
	'CIFAR_Airplane_Automobile':cifar_data[0],
	'CIFAR_Bird_Cat':cifar_data[1],
	'CIFAR_Deer_Dog':cifar_data[2],
	'CIFAR_Frog_Horse':cifar_data[3],
	'CIFAR_Ship_Truck':cifar_data[4],
	'FASHION_MNIST_Shirt_Trouser':fashion_mnist_data[0],
	'FASHION_MNIST_Pullover_Dress':fashion_mnist_data[1],
	'FASHION_MNIST_Coat_Sandal':fashion_mnist_data[2],
	'FASHION_MNIST_Shirt_Sneaker':fashion_mnist_data[3],
	'FASHION_MNIST_Bag_Boot':fashion_mnist_data[4],
	'DIGIT_MNIST_01':digit_mnist_data[0],
	'DIGIT_MNIST_23':digit_mnist_data[1],
	'DIGIT_MNIST_45':digit_mnist_data[2],
	'DIGIT_MNIST_67':digit_mnist_data[3],
	'DIGIT_MNIST_89':digit_mnist_data[4]}
	return data_dict

def load_zero_shot_datasets():
	shvn_data = shvn_preprocessing(shvn_path)
	covertype_data = covertype_preprocessing(covertype_path)
	crop_mapping_data = crop_mapping_preprocessing(crop_mapping_path)
	data_dict = {'SHVN_DATA_01':shvn_data[0],
				'SHVN_DATA_23':shvn_data[1],
				'SHVN_DATA_45':shvn_data[2],
				'SHVN_DATA_67':shvn_data[3],
				'SHVN_DATA_89':shvn_data[4],
				'COVERTYPE_01':covertype_data[0],
				'COVERTYPE_23':covertype_data[1],
				'COVERTYPE_45':covertype_data[2],
				'CROP_MAPPING_01':crop_mapping_data[0],
				'CROP_MAPPING_23':crop_mapping_data[1],
				'CROP_MAPPING_45':crop_mapping_data[2]}
	return data_dict
