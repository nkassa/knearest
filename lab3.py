import numpy as np
from lab3_utils import edit_distance, feature_names
import pandas as pd



# Hint: Consider how to utilize np.unique()
def preprocess_data(training_inputs, testing_inputs, training_labels, testing_labels):
	processed_training_inputs, processed_testing_inputs = ([], [])
	processed_training_labels, processed_testing_labels = ([], [])
	# VVVVV YOUR CODE GOES HERE VVVVV $
################# PART 2 a) #######################################################
	# loop through the column of the training input to find the '?'
	for x in range(training_inputs.shape[1]):
		# we are looping over the columns 
		columns = training_inputs[:, x]
		# unique value
		uniqueval = np.unique(columns)
		# number of times repeated calculated in this loop
		# create array the size of uniqueval
		val_cnt = np.empty_like(uniqueval)
		for i in range(len(uniqueval)):
			# in the array store all the values of the column
			val = uniqueval[i]
			val_cnt[i] = np.count_nonzero(columns == val)
		# return indices of the maximum values (index of most repeated data)
		most_repeated = np.argmax(val_cnt)
		#mode
		mode = uniqueval[most_repeated]
		# replace '?' with mode for training  
		# find the '?' 
		train_question = training_inputs[:, x] == '?'
		# replace '?' with mode
		training_inputs[train_question , x] = mode
		
		
	# loop through the column of the testing input to find the '?'
	for x in range(testing_inputs.shape[1]):
		# we are looping over the columns 
		columns = testing_inputs[:, x]
		# unique value
		uniqueval = np.unique(columns)
		# number of times repeated calculated in this loop
		# create array the size of uniqueval
		val_cnt = np.empty_like(uniqueval)
		for i in range(len(uniqueval)):
			# in the array store all the values of the column
			val = uniqueval[i]
			val_cnt[i] = np.count_nonzero(columns == val)
		# return indices of the maximum values (index of most repeated data)
		most_repeated = np.argmax(val_cnt)
		#mode
		mode = uniqueval[most_repeated]
		# replace '?' with mode for testing 
		# find the '?' 
		test_question = testing_inputs[:, x] == '?'
		# replace '?' with mode
		testing_inputs[test_question , x] = mode
##############################################################################################################
	
######################################################## PART 2 b)############################################
	# map of features names (key is name, value is the order in which they appear in feature_names)
	featurenames = {
    'age_group': 0,
    'menopause': 1,
    'tumor_size': 2,
    'inv_nodes': 3,
    'node_caps': 4,
    'deg_malig': 5,
    'side': 6,
    'quadrant': 7,
    'irradiated': 8
    }
	
	# hard coding the ordinal features based on instructions 
	ordinal = { 
	'age_group':{"10-19": 1, "20-29": 2, "30-39": 3, "40-49": 4, "50-59": 5, "60-69": 6, "70-79": 7, "80-89": 8, "90-99": 9},
    'tumor_size':{"0-4":1, "5-9":2, "10-14":3, "15-19":4, "20-24":5, "25-29":6, "30-34":7, "35-39":8, "40-44":9, "45-49":10, "50-54":11, "55-59":12},
    'inv_nodes':{"0-2": 1, "3-5": 2, "6-8": 3, "9-11": 4, "12-14":5, "15-17":6, "18-20":7, "21-23":8, "24-26":9, "27-29":10, "30-32":11, "33-35":12, "36-39":13},
    'deg_malig':{1: 1,2: 2,3: 3},
	}
	
	# hard coding the categorical 
	categorical = {
	 'irradiated':{"yes": 1, "no": 0},
	 'node_caps':{"yes": 1, "no": 0},
	 'side':{"left": 0, "right": 1},
	}

	# converting ordinal to numeric features 
	# for training and testing
	for feature_indx in ordinal:
		val = ordinal[feature_indx]
		name = featurenames[feature_indx]
		training_inputs[:, name] = np.vectorize(val.get)(training_inputs[:, name])
		testing_inputs[:, name] = np.vectorize(val.get)(testing_inputs[:, name])

	#converting categorical to numeric features
	# for training and testing 
	for feature_indx in categorical:
		val = categorical[feature_indx]
		name = featurenames[feature_indx]
		training_inputs[:, name] = np.vectorize(val.get)(training_inputs[:, name])
		testing_inputs[:, name] = np.vectorize(val.get)(testing_inputs[:, name])
		
################ TRAINING #######################################################
	#one hot for quadrant 
	quadrant_array = ['left_up', 'left_low', 'right_up', 'right_low', 'central']
	for ind, quadrant in enumerate(training_inputs[:,7]):
		array = [0, 0, 0, 0, 0]
		index = quadrant_array.index(quadrant)
		array[index] = 1
		training_inputs[:, 7][ind] = array
	create = []
	for test in training_inputs:
		new_create = []
		for ind, val_test in enumerate(test):
		
			if ind == 7:
				# this is the list
				# loop through it and add to create
				for item in val_test:
					new_create.extend([item])
			else:
				new_create.extend([val_test]) 
		create.append(new_create)
	processed_training_inputs = np.array(create)
	
	
	#one hot for menopause 
	menopause_array = ['ge40', 'lt40', 'premeno']
	for ind, menopause in enumerate(training_inputs[:,1]):
		array = [0, 0, 0]
		index = menopause_array.index(menopause)
		array[index] = 1
		training_inputs[:, 1][ind] = array
	create = []
	for test in training_inputs:
		new_create = []
		for ind, val_test in enumerate(test):
			
			if ind == 1:
				# this is the list
				# loop through it and add to create
				for item in val_test:
					new_create.extend([item])
			elif ind == 7:
				# this is the list
				# loop through it and add to create
				for item in val_test:
					new_create.extend([item])
			else:
				new_create.extend([val_test]) 
		create.append(new_create)
	processed_training_inputs = np.array(create).reshape(-1, len(create[0]))
####################################################################################	
		 
################ TESTING #######################################################
	#one hot for quadrant 
	quadrant_array = ['left_up', 'left_low', 'right_up', 'right_low', 'central']
	for ind, quadrant in enumerate(testing_inputs[:,7]):
		array = [0, 0, 0, 0, 0]
		index = quadrant_array.index(quadrant)
		array[index] = 1
		testing_inputs[:, 7][ind] = array
	create = []
	for test in testing_inputs:
		new_create = []
		for ind, val_test in enumerate(test):
		
			if ind == 7:
				# this is the list
				# loop through it and add to create
				for item in val_test:
					new_create.extend([item])
			else:
				new_create.extend([val_test]) 
		create.append(new_create)
	processed_testing_inputs = np.array(create)
	
	
	#one hot for menopause 
	menopause_array = ['ge40', 'lt40', 'premeno']
	for ind, menopause in enumerate(testing_inputs[:,1]):
		array = [0, 0, 0]
		index = menopause_array.index(menopause)
		array[index] = 1
		testing_inputs[:, 1][ind] = array
	create = []
	for test in testing_inputs:
		new_create = []
		for ind, val_test in enumerate(test):
			
			if ind == 1:
				# this is the list
				# loop through it and add to create
				for item in val_test:
					new_create.extend([item])
			elif ind == 7:
				# this is the list
				# loop through it and add to create
				for item in val_test:
					new_create.extend([item])
			else:
				new_create.extend([val_test]) 
		create.append(new_create)

	processed_testing_inputs = np.array(create).reshape(-1, len(create[0]))
	#print(processed_testing_inputs)
####################################################################################
		
	processed_training_labels = training_labels
	processed_testing_labels = testing_labels
	# ^^^^^ YOUR CODE GOES HERE ^^^^^ $
	return processed_training_inputs, processed_testing_inputs, processed_training_labels, processed_testing_labels

############################################################################################################

####################################################### DISTANCE FUNCTION ######################################
# calculate distance 
def Distance(L, x1 , x2):
	# for L bigger than 1000 (inf)
	if L == np.inf:
		return np.max(np.abs(np.array(x1) - np.array(x2)))
	# for L is -1
	elif L == -1:
		res = edit_distance(x1, x2, L)
		return res
	# for L ->{1,2,3,4,5,6}
	else:
		return np.power(
        np.sum(
            np.power(np.abs(np.array(x1) - np.array(x2)), L)
        ), np.divide(1, L))

##############################################################################################################	

# Hint: consider how to utilize np.argsort()
def k_nearest_neighbors(predict_on, reference_points, reference_labels, k, l, weighted):
    assert len(predict_on) > 0, f"parameter predict_on needs to be of length 0 or greater"
    assert len(reference_points) > 0, f"parameter reference_points needs to be of length 0 or greater"
    assert len(reference_labels) > 0, f"parameter reference_labels needs to be of length 0 or greater"
    assert len(reference_labels) == len(reference_points), f"reference_points and reference_labels need to be the" \
                                                           f" same length"
    predictions = []
    res = []
    # VVVVV YOUR CODE GOES HERE VVVVV $
    
    # iterate over predict_on
    for x1 in range(len(predict_on)):
    	#empty list
    	dist_list = []
    	# calculate distances 
    	for x2 in range(len(reference_points)):
    		dist = Distance(l, predict_on[x1], reference_points[x2])
    		dist_list.append(dist)
    	# sort
    	k_nearest = np.argsort(dist_list)[:k]
    	# get labels of k-nearest neighbors 
    	nearest_lab = reference_labels[k_nearest]
    	#unique labels and their count
    	lab, lab_cnt = np.unique(nearest_lab, return_counts = True)
    	#predict label based on majority
    	pred = lab[0]
    	if len(lab) > 1 and lab_cnt[0] != lab_cnt[1]:
    		best_lab = np.argmax(lab_cnt)
    		pred = lab[best_lab]
    	# break ties 
    	elif len(lab) > 1 and lab_cnt[0] == lab_cnt[1]:
    		pred = 'yes'
    	#append the predicted label to list of predictions 
    	predictions.append(pred)
    	
    # ^^^^^ YOUR CODE GOES HERE ^^^^^ $
    return predictions
