# /usr/bin/python3

import os
import json as JSON
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils import data as torchdata
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pandas import DataFrame, Series
from datetime import datetime as Date
from pymongo import MongoClient

debugMode = False

# Number of features (this likely changes each year) - 6 teams, stdev and mean
n_features_per_team = 8
n_input_features = n_features_per_team * 6 * 2
# Ratio of input features to hidden layers
hidden_layer_ratio = 1.5
two_hidden_layers = False
# Year of events & matches.
year = 2019

def debug(msg):
	if (debugMode == True):
		print(msg)

def main():
	
	init_database()
	
	os.environ['OMP_NUM_THREADS'] = "16"
	print(os.environ['OMP_NUM_THREADS'])
	
	train()
	
	# matches = matchesCol.find({'event_key': '2019paca'})
	# matches_df = DataFrame(matches)
	
	# matches = getMatchHistory('frc41', 2019)
	# getUpcomingMatch('2019mesh_qf1m2', 2019)
	
	

# 	-------------------------------------------------------
#	---					    FRC stuff					---
# 	-------------------------------------------------------

def getMatchesForTrainingData():
	
	max_time = 999999999999
	
	pipeline = [
		{'$match': {
			'event_key': {'$in': event_keys.tolist()},
			'predicted_time': 	{'$ne': None},
			'actual_time': 		{'$ne': None},
			'score_breakdown': 	{'$ne': None},
			'score_breakdown.blue.totalPoints': {'$gte': 0},
			'score_breakdown.red.totalPoints': {'$gte': 0},
			'score_breakdown.blue.autoPoints': {'$gte': 0},
			'score_breakdown.red.autoPoints': {'$gte': 0},
		}},
		{'$project': {
			'_id': 0,
			'key': 1,
			'winner': '$winning_alliance',
		}}
	]
	
	matches = matchesCol.aggregate(pipeline)
	matches = DataFrame(matches)
	
	train_keys = matches.sample(frac=0.7, random_state=453252323)
	validate_keys = matches.drop(train_keys.index)
	
	return matches, train_keys, validate_keys

def createTrainingData(type='train', min=0, max=9999999):
	
	matches, train_keys, validate_keys = getMatchesForTrainingData()
	
	max_time = 999999999999999
	
	if type == 'train':
		existing_data = DataFrame(trainDataCol.find({'type': 'train'}))
		matchlist = train_keys
		print('Inserting training data...')
	elif type == 'validate':
		existing_data = DataFrame(trainDataCol.find({'type': 'validate'}))
		matchlist = validate_keys
		print('Inserting validation data...')
	else:
		print(f'Invalid type {type}')
		exit(1)
	
	if len(existing_data) > 0:
		existing_keylist = existing_data['key'].values
	else:
		existing_keylist = []							# Just in case there's nothing in the db yet
	
	length = len(matchlist.index)
	
	i = 0
	
	for idx, row in matchlist.iterrows():
		key = row['key']
		winner = row['winner']
		# only if we haven't yet calculated
		if (not key in existing_keylist and i >= min and i < max):
			item = getUpcomingMatch(key, year, max_time)
			try:
				json = JSON.loads(item.to_json())
				trainDataCol.insert_one({
					'type': type, 
					'key': key, 
					'data': json, 
					'winner': winner
				})
				print(f'{i:5d} of {length:5d} ({(i / length * 100):2.2f}% complete)   ', end='\r')
			except Exception as e:
				print(e)
				print(item)
				exit(1)
		else: print(f'Skipping {i}                                       ', end='\r')
		i += 1
	print('\ndone')

def wipeTrainingData():
	ans = input('Wipe training data? (type "yes"): ')
	if (ans == 'yes'):
		trainDataCol.delete_many({})
		print('Deleted.')
		exit()

def getUpcomingMatch(match_key, year, max_time=None):
	
	match_details = matchesCol.find_one({'key': match_key})
	time = match_details['predicted_time']
	if max_time: time = max_time	# Manually specified max time
	if not time:
		print(f'Error: No predicted time, {match_key}')
		exit(1)
	
	blue_alliance = match_details['alliances']['blue']['team_keys']
	red_alliance = match_details['alliances']['red']['team_keys']
	
	match_histories_blue = {}
	match_histories_red = {}

	# Dynamically add a prefix or suffix to a series' columns
	def ColumnAppender(columns, prepend, append):
		ret = {}
		for column in columns:
			if (prepend):
				for key in append:
					ret[f'{column}_{key}'] = f'{prepend}_{column}_{key}'
			else:
				ret[column] = f'{column}_{append}'
		return ret

	# DataFrame column renamers
	mapperStd 	= None
	mapperMean 	= None
	mapperBlue1 = None
	mapperBlue2 = None
	mapperBlue3 = None
	mapperRed1 	= None
	mapperRed2 	= None
	mapperRed3 	= None

	std_mean_blue = {}
	std_mean_red = {}

	# for ordering the teams. For now, ordering based on sum of all SELF mean values.
	orders_blue = {}
	orders_red = {}
	
	for team_key in blue_alliance:
		history = getMatchHistory(team_key, year, time)
		if (mapperStd == None):								# Only create the mappers once.... Extremely minor performance thing
			mapperStd = 	ColumnAppender(history.columns, None, 	'std')
			mapperMean = 	ColumnAppender(history.columns, None, 	'mean')
			mapperBlue1 = 	ColumnAppender(history.columns, 'blue1', ['std', 'mean'])
			mapperBlue2 = 	ColumnAppender(history.columns, 'blue2', ['std', 'mean'])
			mapperBlue3 = 	ColumnAppender(history.columns, 'blue3', ['std', 'mean'])
			mapperRed1 = 	ColumnAppender(history.columns, 'red1',  ['std', 'mean'])
			mapperRed2 = 	ColumnAppender(history.columns, 'red2',  ['std', 'mean'])
			mapperRed3 = 	ColumnAppender(history.columns, 'red3',  ['std', 'mean'])
		if (len(history.columns) == 0):
			print(f'Could not find match history for team {team_key}')
			exit(1)
		match_histories_blue[team_key] = history
		std = history.std().rename(mapperStd)
		mean = history.mean().rename(mapperMean)
		std_mean = std.append(mean)
		std_mean_blue[team_key] = std_mean
		orders_blue[team_key] = mean.filter(regex='self').sum()
	for team_key in red_alliance:
		history = getMatchHistory(team_key, year, time)
		match_histories_red[team_key] = history
		std = history.std().rename(mapperStd)
		mean = history.mean().rename(mapperMean)
		std_mean = std.append(mean)
		std_mean_red[team_key] = std_mean
		debug('filter:')
		debug(mean.filter(regex='self'))
		orders_red[team_key] = mean.filter(regex='self').sum()
	
	# Get sorted team keys
	sorted_red = sorted(orders_red, key=lambda team: orders_red[team], reverse=True)
	sorted_blue = sorted(orders_blue, key=lambda team: orders_blue[team], reverse=True)

	debug(mapperBlue1)
	
	debug(sorted_red)
	debug(std_mean_red)

	final_data = pd.Series(dtype=np.float64).append([
		std_mean_blue[sorted_blue[0]].rename(mapperBlue1),
		std_mean_blue[sorted_blue[1]].rename(mapperBlue2),
		std_mean_blue[sorted_blue[2]].rename(mapperBlue3),
		std_mean_red[sorted_red[0]].rename(mapperRed1),
		std_mean_red[sorted_red[1]].rename(mapperRed2),
		std_mean_red[sorted_red[2]].rename(mapperRed3)
	])

	return final_data

def getMatchHistory(team_key, year, max_time):
	
	pipeline = [
		{'$match': {	# 0
			'event_key': {'$in': event_keys.tolist()},		# Events
			'predicted_time': {								# Matches that PRECEDE the match we're examining (for training)
				'$ne': None, 
				'$lt': max_time
			},			
			'$or': [										# Specified team in either red or blue alliance
				{'alliances.blue.team_keys': team_key},
				{'alliances.red.team_keys': team_key},
			],
			'actual_time': 		{'$ne': None}, 				# to avoid matches that didn't occur
			'score_breakdown': 	{'$ne': None}, 				# to avoid broken data
			'score_breakdown.blue.totalPoints': {'$gte': 0},
			'score_breakdown.red.totalPoints': {'$gte': 0},
			'score_breakdown.blue.autoPoints': {'$gte': 0},
			'score_breakdown.red.autoPoints': {'$gte': 0},
		}},
		{'$project': {	# 1
			'_id': 				0,
			'alliance_blue': 	'$alliances.blue.team_keys',
			'alliance_red': 	'$alliances.red.team_keys',
			'blue': 			'$score_breakdown.blue',
			'red': 				'$score_breakdown.red',
			'color': {
				'$cond': [{'$in': [team_key, '$alliances.blue.team_keys']}, 'blue', 'red']
			},
			'winner': 			'$winning_alliance',
			'key': 				1, 							# for debugging
			'predicted_time':	1,
		}},
		{'$project': {	# 2
			'key': 				1, 							# for debugging
			'predicted_time':	1,
			'color':			1,
			'alliance_self': {
				'$cond': [{'$eq': ['$color', 'blue']}, '$alliance_blue', '$alliance_red']
			},
			'alliance_opp': {
				'$cond': [{'$eq': ['$color', 'blue']}, '$alliance_red', '$alliance_blue']
			},
			'score_self': {
				'$cond': [{'$eq': ['$color', 'blue']}, '$blue', '$red']
			},
			'score_opp': {
				'$cond': [{'$eq': ['$color', 'blue']}, '$red', '$blue']
			},
			'winner': 1,
		}},
		{'$project': {	# 3
			# 'key': 				1, 							# for debugging
			# 'predicted_time':	1,
			# 'color':			1,
			# 'winner':			1,
			# 'alliance_self': 	1,
			# 'alliance_opp': 	1,
			# did_win: switch below
			# 'cargo_self': 		'$score_self.cargoPoints',
			# 'cargo_opp': 		'$score_opp.cargoPoints',
			# 'panel_self': 		'$score_self.hatchPanelPoints',
			# 'panel_opp': 		'$score_opp.hatchPanelPoints',
			# 'habclimb_self':	'$score_self.habClimbPoints',
			# 'habclimb_opp':		'$score_opp.habClimbPoints',
			'auto_self': 		'$score_self.autoPoints',
			'auto_opp': 		'$score_opp.autoPoints',
			# Total points = cargoPoints + panelPoints + autoPoints + habClimbPoints + foulPoints, so not needed
			'teleop_self':		'$score_self.teleopPoints',
			'teleop_opp':		'$score_opp.teleopPoints',
			# 'foul_self':		'$score_self.foulPoints',
			# 'foul_opp':			'$score_opp.foulPoints',
		}},
	]
	# win: 1, tie: 0.5, loss: 0
	pipeline[3]['$project']['did_win'] = MongoSwitch('$winner', [['$color', ''], [1, 0.5], 0])
	
	dtFinishedPipeline = Date.now()
	
	matches = matchesCol.aggregate(pipeline)
	
	return DataFrame(matches)

def getEventsByYear(year):
	
	eventsCol = db['events']
	
	events = eventsCol.find({'year': year})
	events_df = DataFrame(events)
	
	keys = events_df['key']
	return keys


# 	-------------------------------------------------------
#	---					    Database					---
# 	-------------------------------------------------------

def init_database():

	global db, matchesCol, eventsCol, trainDataCol, event_keys
	
	st = Date.now()
	
	db = get_database()
	matchesCol = db['matches']
	trainDataCol = db['trainingdata']
	
	event_keys = getEventsByYear(year)

def get_database():

	# Provide the mongodb atlas url to connect python to mongodb using pymongo
	CONNECTION_STRING = "mongodb://localhost:27017/app"
	# CONNECTION_STRING = "mongodb://DbUser:v5R4AYRsNrKC7jQq@scoutradioz-shard-00-00-cpcvb.mongodb.net:27017,scoutradioz-shard-00-01-cpcvb.mongodb.net:27017,scoutradioz-shard-00-02-cpcvb.mongodb.net:27017/prod?authSource=admin&gssapiServiceName=mongodb&replicaSet=Scoutradioz-shard-0&retryWrites=true&ssl=true&w=majority"
	
	client = MongoClient(CONNECTION_STRING)

	# Create the database for our example (we will use the same database throughout the tutorial
	return client['app']

# MongoSwitch($'myField', [['$item1', '$item2', '$item3'], [value1, value2, value3], default])
def MongoSwitch(field, params):
	origValues = params[0]
	destValues = params[1]
	default = params[2]
	
	switch = {
		'$switch': {
			'branches': [],
			'default': default
		}
	}
	for i in range(len(origValues)):
		switch['$switch']['branches'].append({
			'case': {'$eq': [field, origValues[i]]},
			'then': destValues[i]
		})
	return switch


# 	-------------------------------------------------------
#	---					Neural Network					---
# 	-------------------------------------------------------

def getTrainingData(type):
	# Pull from database
	dataframe = DataFrame(trainDataCol.find({'type': type}))
	data = dataframe['data']
	
	# Turn into a Numpy array (removing labels in the process)
	# format: [feature1, ... featureN, blueWon, redWon]
	num_entries = len(data.index)
	data_arr = np.empty([num_entries, n_input_features + 2], dtype=float)
	
	for i, row in dataframe.iterrows():
		features = np.fromiter(row['data'].values(), dtype=float)
		winner = row['winner']
		if (winner == 'blue'):
			label = np.array([1, 0], dtype=float)
		elif (winner == 'red'):
			label = np.array([0, 1], dtype=float)
		else:
			label = np.array([0.5, 0.5], dtype=float)
		data_arr[i] = np.concatenate((features, label))
	
	# Turn into a Torch dataset
	dataset = torch.tensor(data_arr, dtype=torch.float32)
	loader = DataLoader(dataset, num_workers=6)
	return loader

def train():
	# Set fixed random number seed
	pl.seed_everything(42)
	
	mlp = MLP()
	trainer = pl.Trainer(gpus=0, max_epochs=5, limit_train_batches=0.5, limit_val_batches=0.5, profiler="simple")
	trainer.fit(mlp)
	
	

class MLP(pl.LightningModule):

	def __init__(self):

		numLayer1 = round(n_input_features * hidden_layer_ratio)
		numLayer2 = round(n_input_features * hidden_layer_ratio * 0.3)

		super().__init__()
		if two_hidden_layers:
			self.layers = nn.Sequential(
				nn.Linear(n_input_features, numLayer1),
				nn.Sigmoid(),
				nn.Linear(numLayer1, numLayer2),
				nn.Sigmoid(),
				nn.Linear(numLayer2, 2),
			)
			print(f'Hidden layer 1: {numLayer1}, Hidden layer 2: {numLayer2}')
		else:
			self.layers = nn.Sequential(
				nn.Linear(n_input_features, numLayer1),
				nn.Sigmoid(),
				nn.Linear(numLayer1, 2),
			)
			print(f'Hidden layer: {numLayer1} neurons')
		self.loss = nn.L1Loss()
	
	def forward(self, x):
		return self.layers(x)
		
	def train_dataloader(self):
		loader = getTrainingData('train')
		return loader
	
	def val_dataloader(self):
		return getTrainingData('validate')
	
	# TODO make this specific to frc
	def training_step(self, batch, batch_idx):
		# format: [feature1, ... featureN, blueWon, redWon]
		features = 		batch[0, 0 : n_input_features]
		label = 		batch[0, n_input_features : n_input_features + 2]
		prediction = 	self.layers(features)
		loss = 			self.loss(prediction, label)
		self.log('train_loss', loss)
		return loss
	
	def validation_step(self, batch, batch_idx):
		features = 		batch[0, 0 : n_input_features]
		label = 		batch[0, n_input_features : n_input_features + 2]
		prediction = 	self.layers(features)
		loss = 			self.loss(prediction, label)
		self.log('val_loss', loss)
		return loss, prediction, label

	def validation_epoch_end(self, validation_step_outputs):
		sum = 0
		i = 0
		nCorrect = 0
		nWrong = 0
		for itm in validation_step_outputs:
			loss, prediction, label = itm
			diff = abs(prediction[0] - prediction[1])
			# Match is a tie
			if (diff < 0.05 and label[0] == 0.5):
				# print(f'Tie: {prediction}')
				nCorrect += 1
			# Blue won!
			elif (prediction[0] > prediction[1] and label[0] > label[1]):
				# print(f'Blue: {prediction}')
				nCorrect += 1
			# Red won!
			elif (prediction[0] < prediction[1] and label[0] < label[1]):
				# print(f'Red: {prediction}')
				nCorrect += 1
			else:
				nWrong += 1
			sum += loss
			i += 1
			# print(f'Label: {label.tolist()}, Prediction: {prediction.tolist()}')
		print(f'\n\n\nLoss: {(sum / i):2.2f}, # correct: {nCorrect}, # wrong: {nWrong}, accuracy: {(100 * nCorrect / (nCorrect + nWrong)):2.2f}%\n')
	
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
		return optimizer

if __name__ == '__main__':
	main()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	# /usr/bin/python3

import os
import json as JSON
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.utils.data as torchdata
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pandas import DataFrame, Series
from datetime import datetime as Date
from pymongo import MongoClient
from typing import Any, Callable, Match, Optional, Tuple

debugMode = False

# Number of features (this likely changes each year) - 6 teams, stdev and mean
n_features_per_team = 9
n_input_features = n_features_per_team * 6 * 2
# Ratio of input features to hidden layers
hidden_layer_ratio = 30
two_hidden_layers = False
# Year of events & matches.
year = 2019

def debug(msg):
	if (debugMode == True):
		print(msg)

def main():
	
	init_database()
	
	train()
	
	# matches = dbMatches.find({'event_key': '2019paca'})
	# matches_df = DataFrame(matches)
	
	# matches = getMatchHistory('frc41', 2019)
	# getUpcomingMatch('2019mesh_qf1m2', 2019)
	
	

# 	-------------------------------------------------------
#	---					    FRC stuff					---
# 	-------------------------------------------------------

def getMatchesForTrainingData():
	
	max_time = 999999999999
	
	pipeline = [
		{'$match': {
			'event_key': {'$in': event_keys.tolist()},
			'predicted_time': 	{'$ne': None},
			'actual_time': 		{'$ne': None},
			'score_breakdown': 	{'$ne': None},
			'score_breakdown.blue.totalPoints': {'$gte': 0},
			'score_breakdown.red.totalPoints': {'$gte': 0},
			'score_breakdown.blue.autoPoints': {'$gte': 0},
			'score_breakdown.red.autoPoints': {'$gte': 0},
		}},
		{'$project': {
			'_id': 0,
			'key': 1,
			'winner': '$winning_alliance',
		}}
	]
	
	matches = dbMatches.aggregate(pipeline)
	matches = DataFrame(matches)
	
	train_keys = matches.sample(frac=0.7, random_state=453252323)
	validate_keys = matches.drop(train_keys.index)
	
	return matches, train_keys, validate_keys

def createTrainingData(type='train', min=0, max=9999999):
	
	matches, train_keys, validate_keys = getMatchesForTrainingData()
	
	max_time = 999999999999999
	
	if type == 'train':
		existing_data = DataFrame(dbTrainData.find({'type': 'train'}))
		matchlist = train_keys
		print('Inserting training data...')
	elif type == 'validate':
		existing_data = DataFrame(dbTrainData.find({'type': 'validate'}))
		matchlist = validate_keys
		print('Inserting validation data...')
	else:
		print(f'Invalid type {type}')
		exit(1)
	
	if len(existing_data) > 0:
		existing_keylist = existing_data['key'].values
	else:
		existing_keylist = []							# Just in case there's nothing in the db yet
	
	length = len(matchlist.index)
	
	items = []
	winners = []
	keys = []
	i = 0
	
	# Retrieve matches
	for idx, row in matchlist.iterrows():
		key = row['key']
		winner = row['winner']
		# only if we haven't yet calculated
		if (not key in existing_keylist and i >= min and i < max):
			item = getUpcomingMatch(key, year, max_time)
			items.append(item.to_dict())
			winners.append(winner)
			keys.append(key)
			print(f'Retrieved {i:5d} of {length:5d} ({(i / length * 100):2.2f}% complete)   ', end='\r')
		else: print(f'Skipping {i}                                       ', end='\r')
		i += 1
	df = DataFrame(items)
	
	# Lookup for weights
	dictLookup = {}
	for col in df.columns:
		max = df[col].abs().max()
		dictLookup[col] = max
		df[col] = df[col] / max
	print(dictLookup)
	
	# Create JSON documents
	modified_items = []
	for idx, row in df.iterrows():
		json = JSON.loads(row.to_json())
		key = keys[idx]
		winner = winners[idx]
		modified_items.append({
			'type': type,
			'key': key,
			'data': json,
			'winner': winner
		})
	
	# Insert into database
	print(f'Inserting documents...')
	dbTrainData.insert_many(modified_items)
	print(f'Done')
	

def wipeTrainingData():
	ans = input('Wipe training data? (type "yes"): ')
	if (ans == 'yes'):
		dbTrainData.delete_many({})
		print('Deleted.')
		exit()

def getUpcomingMatch(match_key, year, max_time=None):
	
	match_details = dbMatches.find_one({'key': match_key})
	time = match_details['predicted_time']
	if max_time: time = max_time	# Manually specified max time
	if not time:
		print(f'Error: No predicted time, {match_key}')
		exit(1)
	
	blue_alliance = match_details['alliances']['blue']['team_keys']
	red_alliance = match_details['alliances']['red']['team_keys']
	
	match_histories_blue = {}
	match_histories_red = {}

	# Dynamically add a prefix or suffix to a series' columns
	def ColumnAppender(columns, prepend, append):
		ret = {}
		for column in columns:
			if (prepend):
				for key in append:
					ret[f'{column}_{key}'] = f'{prepend}_{column}_{key}'
			else:
				ret[column] = f'{column}_{append}'
		return ret

	# DataFrame column renamers
	mapperStd 	= None
	mapperMean 	= None
	mapperBlue1 = None
	mapperBlue2 = None
	mapperBlue3 = None
	mapperRed1 	= None
	mapperRed2 	= None
	mapperRed3 	= None

	std_mean_blue = {}
	std_mean_red = {}

	# for ordering the teams. For now, ordering based on sum of all SELF mean values.
	orders_blue = {}
	orders_red = {}
	
	for team_key in blue_alliance:
		history = getMatchHistory(team_key, year, time)
		if (mapperStd == None):								# Only create the mappers once.... Extremely minor performance thing
			mapperStd = 	ColumnAppender(history.columns, None, 	'std')
			mapperMean = 	ColumnAppender(history.columns, None, 	'mean')
			mapperBlue1 = 	ColumnAppender(history.columns, 'blue1', ['std', 'mean'])
			mapperBlue2 = 	ColumnAppender(history.columns, 'blue2', ['std', 'mean'])
			mapperBlue3 = 	ColumnAppender(history.columns, 'blue3', ['std', 'mean'])
			mapperRed1 = 	ColumnAppender(history.columns, 'red1',  ['std', 'mean'])
			mapperRed2 = 	ColumnAppender(history.columns, 'red2',  ['std', 'mean'])
			mapperRed3 = 	ColumnAppender(history.columns, 'red3',  ['std', 'mean'])
		if (len(history.columns) == 0):
			print(f'Could not find match history for team {team_key}')
			exit(1)
		match_histories_blue[team_key] = history
		std = history.std().rename(mapperStd)
		mean = history.mean().rename(mapperMean)
		std_mean = std.append(mean)
		std_mean_blue[team_key] = std_mean
		orders_blue[team_key] = mean.filter(regex='self').sum()
	for team_key in red_alliance:
		history = getMatchHistory(team_key, year, time)
		match_histories_red[team_key] = history
		std = history.std().rename(mapperStd)
		mean = history.mean().rename(mapperMean)
		std_mean = std.append(mean)
		std_mean_red[team_key] = std_mean
		debug('filter:')
		debug(mean.filter(regex='self'))
		orders_red[team_key] = mean.filter(regex='self').sum()
	
	# Get sorted team keys
	sorted_red = sorted(orders_red, key=lambda team: orders_red[team], reverse=True)
	sorted_blue = sorted(orders_blue, key=lambda team: orders_blue[team], reverse=True)

	debug(mapperBlue1)
	
	debug(sorted_red)
	debug(std_mean_red)

	final_data = pd.Series(dtype=np.float64).append([
		std_mean_blue[sorted_blue[0]].rename(mapperBlue1),
		std_mean_blue[sorted_blue[1]].rename(mapperBlue2),
		std_mean_blue[sorted_blue[2]].rename(mapperBlue3),
		std_mean_red[sorted_red[0]].rename(mapperRed1),
		std_mean_red[sorted_red[1]].rename(mapperRed2),
		std_mean_red[sorted_red[2]].rename(mapperRed3)
	])

	return final_data

def getMatchHistory(team_key, year, max_time):
	
	pipeline = [
		{'$match': {	# 0
			'event_key': {'$in': event_keys.tolist()},		# Events
			'predicted_time': {								# Matches that PRECEDE the match we're examining (for training)
				'$ne': None, 
				'$lt': max_time
			},			
			'$or': [										# Specified team in either red or blue alliance
				{'alliances.blue.team_keys': team_key},
				{'alliances.red.team_keys': team_key},
			],
			'actual_time': 		{'$ne': None}, 				# to avoid matches that didn't occur
			'score_breakdown': 	{'$ne': None}, 				# to avoid broken data
			'score_breakdown.blue.totalPoints': {'$gte': 0},
			'score_breakdown.red.totalPoints': {'$gte': 0},
			'score_breakdown.blue.autoPoints': {'$gte': 0},
			'score_breakdown.red.autoPoints': {'$gte': 0},
		}},
		{'$project': {	# 1
			'_id': 				0,
			'alliance_blue': 	'$alliances.blue.team_keys',
			'alliance_red': 	'$alliances.red.team_keys',
			'blue': 			'$score_breakdown.blue',
			'red': 				'$score_breakdown.red',
			'color': {
				'$cond': [{'$in': [team_key, '$alliances.blue.team_keys']}, 'blue', 'red']
			},
			'winner': 			'$winning_alliance',
			'key': 				1, 							# for debugging
			'predicted_time':	1,
		}},
		{'$project': {	# 2
			'key': 				1, 							# for debugging
			'predicted_time':	1,
			'color':			1,
			'alliance_self': {
				'$cond': [{'$eq': ['$color', 'blue']}, '$alliance_blue', '$alliance_red']
			},
			'alliance_opp': {
				'$cond': [{'$eq': ['$color', 'blue']}, '$alliance_red', '$alliance_blue']
			},
			'score_self': {
				'$cond': [{'$eq': ['$color', 'blue']}, '$blue', '$red']
			},
			'score_opp': {
				'$cond': [{'$eq': ['$color', 'blue']}, '$red', '$blue']
			},
			'winner': 1,
		}},
		{'$project': {	# 3
			# 'key': 				1, 							# for debugging
			# 'predicted_time':	1,
			# 'color':			1,
			# 'winner':			1,
			# 'alliance_self': 	1,
			# 'alliance_opp': 	1,
			# did_win: switch below
			'cargo_self': 		'$score_self.cargoPoints',
			'cargo_opp': 		'$score_opp.cargoPoints',
			'panel_self': 		'$score_self.hatchPanelPoints',
			'panel_opp': 		'$score_opp.hatchPanelPoints',
			'habclimb_self':	'$score_self.habClimbPoints',
			'habclimb_opp':		'$score_opp.habClimbPoints',
			'auto_self': 		'$score_self.autoPoints',
			'auto_opp': 		'$score_opp.autoPoints',
			# Total points = cargoPoints + panelPoints + autoPoints + habClimbPoints + foulPoints, so not needed
			# 'teleop_self':		'$score_self.teleopPoints',
			# 'teleop_opp':		'$score_opp.teleopPoints',
			# 'foul_self':		'$score_self.foulPoints',
			# 'foul_opp':			'$score_opp.foulPoints',
		}},
	]
	# win: 1, tie: 0.5, loss: 0
	pipeline[3]['$project']['did_win'] = MongoSwitch('$winner', [['$color', ''], [1, 0.5], 0])
	
	dtFinishedPipeline = Date.now()
	
	matches = dbMatches.aggregate(pipeline)
	
	return DataFrame(matches)

def getEventsByYear(year):
	
	dbEvents = db['events']
	
	events = dbEvents.find({'year': year})
	events_df = DataFrame(events)
	
	keys = events_df['key']
	return keys


# 	-------------------------------------------------------
#	---					    Database					---
# 	-------------------------------------------------------

def init_database():

	global db, dbMatches, dbEvents, dbTrainData, dbOPR, event_keys
	
	st = Date.now()
	
	db = get_database()
	dbMatches = db['matches']
	dbTrainData = db['trainingdata']
	dbOPR = db['oprs']
	
	event_keys = getEventsByYear(year)

def get_database():

	# Provide the mongodb atlas url to connect python to mongodb using pymongo
	CONNECTION_STRING = "mongodb://localhost:27017/app"
	# CONNECTION_STRING = "mongodb://DbUser:v5R4AYRsNrKC7jQq@scoutradioz-shard-00-00-cpcvb.mongodb.net:27017,scoutradioz-shard-00-01-cpcvb.mongodb.net:27017,scoutradioz-shard-00-02-cpcvb.mongodb.net:27017/prod?authSource=admin&gssapiServiceName=mongodb&replicaSet=Scoutradioz-shard-0&retryWrites=true&ssl=true&w=majority"
	
	client = MongoClient(CONNECTION_STRING)

	# Create the database for our example (we will use the same database throughout the tutorial
	return client['app']

# MongoSwitch($'myField', [['$item1', '$item2', '$item3'], [value1, value2, value3], default])
def MongoSwitch(field, params):
	origValues = params[0]
	destValues = params[1]
	default = params[2]
	
	switch = {
		'$switch': {
			'branches': [],
			'default': default
		}
	}
	for i in range(len(origValues)):
		switch['$switch']['branches'].append({
			'case': {'$eq': [field, origValues[i]]},
			'then': destValues[i]
		})
	return switch


# 	-------------------------------------------------------
#	---					Neural Network					---
# 	-------------------------------------------------------

def getTrainingData(type):
	# Pull from database
	trainFrame = DataFrame(dbTrainData.find({'type': type}))
	
	if (len(trainFrame.index) == 0):
		print(f'{type} data does not exist')
		exit(1)
	data = trainFrame['data']
	
	# Turn into a Numpy array (removing labels in the process)
	# format: [feature1, ... featureN, blueWon, redWon]
	n_entries = len(data.index)
	
	data_arr = np.empty([n_entries, n_input_features], dtype=float)
	label_arr = np.empty([n_entries], dtype=int)
	keys_arr = []
	
	last_idx = 0
	for i, row in trainFrame.iterrows():
		# Sanity check
		if (last_idx != i): 
			print('Indexing error')
			exit(1)
		last_idx += 1
		
		features = np.fromiter(row['data'].values(), dtype=float)
		winner = row['winner']
		if (winner == 'blue'):
			label = 0
		elif (winner == 'red'):
			label = 2
		else:
			label = 1
		data_arr[i] = features
		label_arr[i] = label
		keys_arr.append(row['key'])
	
	dataset = MatchDataset()
	dataset.populate(data_arr, label_arr, keys_arr, n_entries)
	
	# Place into Torch DataLoader
	loader = DataLoader(dataset, batch_size=128)
	return loader

class OPRComparison():
	
	def __init__(self):
		self.oprs, self.dprs = getOPRs()
	
	def getPrediction(self, match_key):
		match     = dbMatches.find_one({'key': match_key})
		event_key = match['event_key']
		bluekeys  = match['alliances']['blue']['team_keys']
		redkeys   = match['alliances']['red']['team_keys']
		
		doCountDPR  = True # Subtract DPR from opposite team
		blueOPR 	= 0
		redOPR   	= 0
		
		# Avoid OPR not predicting anything
		if not event_key in self.oprs: return None
		
		eventOPRs = self.oprs[event_key]
		eventDPRs = self.dprs[event_key]
		for team_key in bluekeys:
			if not team_key in eventOPRs: return None
			blueOPR += eventOPRs[team_key]
			if doCountDPR: redOPR -= eventDPRs[team_key]
		for team_key in redkeys:
			if not team_key in eventOPRs: return None
			redOPR  += eventOPRs[team_key]
			if doCountDPR: blueOPR -= eventDPRs[team_key]
		
		if (abs(blueOPR - redOPR) < 0.05): 	return 1
		elif (blueOPR > redOPR):		 	return 0
		else:								return 2

# OPRs for comparison
def getOPRs():
	
	pipeline = [
		{'$match': {
			'oprs': {'$ne': None},
			'dprs': {'$ne': None},
		}},
		{'$sort': {
			'event_key': 1,
		}},
		{'$project': {
			'_id': 			0,
			'event_key': 	1,
			'oprs':			1,
			'dprs':			1,
		}}
	]
	oprFrame = DataFrame(dbOPR.aggregate(pipeline))
	
	oprDict = {}
	dprDict = {}
	
	for i, row in oprFrame.iterrows():
		key = row['event_key']
		opr = row['dprs']
		dpr = row['oprs']
		oprDict[key] = opr
		dprDict[key] = dpr
	
	return oprDict, dprDict

def train():
	# Set fixed random number seed
	pl.seed_everything(42)
	
	mlp = MLP()
	trainer = pl.Trainer(
		gpus=1, 
		max_epochs=500, 
		check_val_every_n_epoch=10,
		limit_train_batches=1.0, 
		limit_val_batches=1.0, 
		auto_lr_find=True,
		enable_checkpointing=False,
		precision=16,
	)
	trainer.tune(mlp)
	trainer.fit(mlp)

# Torch-based dataset to be wrapped into a DataLoader. This helps reduce memory issues.
class MatchDataset(Dataset):
	
	def __init__(self):
		return
	
	def __len__(self):
		return self.length
	
	def __getitem__(self, index: int):
		return self.features[index], self.labels[index], self.keys[index]
	
	def populate(self, features, labels, keys, length):
		st = Date.now()
		self.features = torch.tensor(features, dtype=torch.float32)
		self.labels = torch.tensor(labels, dtype=torch.long)
		self.keys = keys
		self.length = length

class MLP(pl.LightningModule):

	def __init__(self):

		numLayer1 = round(n_input_features * hidden_layer_ratio)
		numLayer2 = round(n_input_features * hidden_layer_ratio * 0.1)
		
		activation = nn.ReLU

		super().__init__()
		if two_hidden_layers:
			self.layers = nn.Sequential(
				nn.Linear(n_input_features, numLayer1),
				activation(),
				nn.Linear(numLayer1, numLayer2),
				activation(),
				nn.Linear(numLayer2, 3),
			)
			print(f'Hidden layer 1: {numLayer1}, Hidden layer 2: {numLayer2}')
		else:
			self.layers = nn.Sequential(
				nn.Linear(n_input_features, numLayer1),
				activation(),
				nn.Linear(numLayer1, 3),
			)
			print(f'Hidden layer: {numLayer1} neurons')
		self.loss = nn.CrossEntropyLoss()	# Loss function
		self.softmax = nn.Softmax(dim=1)	# Softmax for identifying prediction
		self.learning_rate = 1e-4
		self.OPR = OPRComparison()
	
	def forward(self, x):
		return self.layers(x)
		
	def train_dataloader(self):
		loader = getTrainingData('train')
		# self.train_labels = labels
		return loader
	
	def val_dataloader(self):
		loader = getTrainingData('validate')
		# self.val_labels = labels
		return loader
	
	def training_step(self, batch, batch_idx):
		features, label, match_key = batch
		prediction = 	self.layers(features)
		loss = 			self.loss(prediction, label)
		self.log('train_loss', loss)
		return loss
	
	def validation_step(self, batch, batch_idx):
		features, label, match_key = batch
		prediction = 	self.layers(features)
		loss = 			self.loss(prediction, label)
		self.log('val_loss', loss)
		return loss, prediction, label, match_key

	def validation_epoch_end(self, validation_step_outputs):
		sum 		= 0
		i 			= 0
		nCorrect 	= 0
		nWrong 		= 0
		nOPRCorrect = 0
		nOPRWrong	= 0
		
		for itm in validation_step_outputs:
			loss, prediction, label, match_keys = itm
			# NN outputs the raw (unnormalized) values; softmax outputs probabilities of each
			probabilities = self.softmax(prediction)
			# Batches of larger than 1 need to iterate
			for j in range(len(probabilities)):
				outcome = label[j]
				guess = probabilities[j]
				match_key = match_keys[j] # for opr checking
				# Blue won
				if outcome == 0:
					if guess[0] - guess[2] > 0.05:			nCorrect += 1
					else:									nWrong += 1
				# Red won
				elif outcome == 2:
					if guess[2] - guess[0] > 0.05:			nCorrect += 1
					else:									nWrong += 1
				# Tie
				elif outcome == 1:
					if abs(guess[0] - guess[2] < 0.05): 	nCorrect += 1
					else:									nWrong += 1
				# OPR prediction
				oprPrediction = self.OPR.getPrediction(match_key)
				if oprPrediction is None: 			pass
				elif oprPrediction == outcome:		nOPRCorrect += 1
				else:								nOPRWrong += 1
				j += 1
			i += 1
			sum += loss
		print(f'\n\n\nLoss: {(sum / i):2.2f}, # correct: {nCorrect}, # wrong: {nWrong}, accuracy: {(100 * nCorrect / (nCorrect + nWrong)):2.2f}% \
			OPR: {nOPRCorrect} vs. {nOPRWrong} correct, accuracy: {(100 * nOPRCorrect / (nOPRCorrect + nOPRWrong))}%')
		print(self.learning_rate)
		print()
	
	def configure_optimizers(self):
		optimizer = torch.optim.NAdam(self.parameters())
		return optimizer


dictLookup = {
	'blue1_cargo_self_std': 	13.853594404416418,
	'blue1_cargo_opp_std': 		15.081930094806014,
	'blue1_panel_self_std': 	9.44017389230109,
	'blue1_panel_opp_std': 		9.648363026488436,
	'blue1_habclimb_self_std': 	10.090511436623597,
	'blue1_habclimb_opp_std': 	10.392304845413264,
	'blue1_auto_self_std': 		6.456331042187644,
	'blue1_auto_opp_std': 		6.549206460568937,
	'blue1_did_win_std': 		0.5345224838248488,
	'blue1_cargo_self_mean': 	37.44642857142857,
	'blue1_cargo_opp_mean': 	32.76315789473684,
	'blue1_panel_self_mean': 	21.3,
	'blue1_panel_opp_mean': 	15.833333333333334,
	'blue1_habclimb_self_mean':	21.785714285714285,
	'blue1_habclimb_opp_mean': 	19.0,
	'blue1_auto_self_mean': 	15.0,
	'blue1_auto_opp_mean': 		14.555555555555555,
	'blue1_did_win_mean': 		0.9196428571428571,
	'blue2_cargo_self_std': 	13.853594404416418,
	'blue2_cargo_opp_std': 		15.081930094806014,
	'blue2_panel_self_std': 	9.44017389230109,
	'blue2_panel_opp_std': 		9.648363026488436,
	'blue2_habclimb_self_std': 	10.090511436623597,
	'blue2_habclimb_opp_std': 	10.392304845413264,
	'blue2_auto_self_std': 		6.4975519865701585,
	'blue2_auto_opp_std': 		6.549206460568937,
	'blue2_did_win_std': 		0.5477225575051662,
	'blue2_cargo_self_mean': 	33.36842105263158,
	'blue2_cargo_opp_mean': 	31.178571428571427,
	'blue2_panel_self_mean': 	21.3,
	'blue2_panel_opp_mean': 	15.833333333333334,
	'blue2_habclimb_self_mean':	20.235294117647058,
	'blue2_habclimb_opp_mean': 	19.0,
	'blue2_auto_self_mean': 	15.0,
	'blue2_auto_opp_mean': 		14.7,
	'blue2_did_win_mean': 		0.9196428571428571,
	'blue3_cargo_self_std': 	13.715250931284782,
	'blue3_cargo_opp_std': 		13.836710066144533,
	'blue3_panel_self_std': 	9.16940642486727,
	'blue3_panel_opp_std': 		8.573883075080696,
	'blue3_habclimb_self_std': 	10.090511436623597,
	'blue3_habclimb_opp_std': 	10.392304845413264,
	'blue3_auto_self_std': 		6.571781469914507,
	'blue3_auto_opp_std': 		6.549206460568937,
	'blue3_did_win_std': 		0.5477225575051662,
	'blue3_cargo_self_mean': 	30.88888888888889,
	'blue3_cargo_opp_mean': 	30.50943396226415,
	'blue3_panel_self_mean': 	16.962962962962962,
	'blue3_panel_opp_mean': 	15.8,
	'blue3_habclimb_self_mean':	20.235294117647058,
	'blue3_habclimb_opp_mean': 	19.0,
	'blue3_auto_self_mean': 	15.0,
	'blue3_auto_opp_mean': 		14.454545454545455,
	'blue3_did_win_mean': 		0.8478260869565217,
	'red1_cargo_self_std': 		13.715250931284782,
	'red1_cargo_opp_std': 		15.081930094806014,
	'red1_panel_self_std': 		9.16940642486727,
	'red1_panel_opp_std': 		9.648363026488436,
	'red1_habclimb_self_std': 	8.740709353364863,
	'red1_habclimb_opp_std': 	9.512670975646593,
	'red1_auto_self_std': 		6.4975519865701585,
	'red1_auto_opp_std': 		5.70087712549569,
	'red1_did_win_std': 		0.5345224838248488,
	'red1_cargo_self_mean': 	37.44642857142857,
	'red1_cargo_opp_mean': 		32.76315789473684,
	'red1_panel_self_mean': 	21.3,
	'red1_panel_opp_mean': 		15.833333333333334,
	'red1_habclimb_self_mean': 	21.785714285714285,
	'red1_habclimb_opp_mean': 	19.0,
	'red1_auto_self_mean': 		15.0,
	'red1_auto_opp_mean': 		14.555555555555555,
	'red1_did_win_mean': 		0.9196428571428571,
	'red2_cargo_self_std': 		13.853594404416418,
	'red2_cargo_opp_std': 		15.081930094806014,
	'red2_panel_self_std': 		9.44017389230109,
	'red2_panel_opp_std': 		9.648363026488436,
	'red2_habclimb_self_std': 	9.810708435174293,
	'red2_habclimb_opp_std': 	9.512670975646593,
	'red2_auto_self_std': 		6.4975519865701585,
	'red2_auto_opp_std': 		5.766281297335398,
	'red2_did_win_std': 		0.5345224838248488,
	'red2_cargo_self_mean': 	34.275,
	'red2_cargo_opp_mean': 		31.178571428571427,
	'red2_panel_self_mean': 	18.873563218390803,
	'red2_panel_opp_mean': 		15.833333333333334,
	'red2_habclimb_self_mean': 	20.0,
	'red2_habclimb_opp_mean': 	19.0,
	'red2_auto_self_mean': 		15.0,
	'red2_auto_opp_mean': 		14.7,
	'red2_did_win_mean': 		0.9196428571428571,
	'red3_cargo_self_std': 		13.853594404416418,
	'red3_cargo_opp_std': 		15.081930094806014,
	'red3_panel_self_std': 		9.44017389230109,
	'red3_panel_opp_std': 		8.573883075080696,
	'red3_habclimb_self_std': 	10.090511436623597,
	'red3_habclimb_opp_std': 	10.392304845413264,
	'red3_auto_self_std': 		6.571781469914507,
	'red3_auto_opp_std': 		6.549206460568937,
	'red3_did_win_std': 		0.5477225575051662,
	'red3_cargo_self_mean': 	31.125,
	'red3_cargo_opp_mean': 		29.785714285714285,
	'red3_panel_self_mean':		16.846153846153847,
	'red3_panel_opp_mean': 		15.833333333333334,
	'red3_habclimb_self_mean': 	20.235294117647058,
	'red3_habclimb_opp_mean': 	19.0,
	'red3_auto_self_mean': 		15.0,
	'red3_auto_opp_mean': 		14.454545454545455,
	'red3_did_win_mean': 		0.7777777777777778
}

if __name__ == '__main__':
	main()