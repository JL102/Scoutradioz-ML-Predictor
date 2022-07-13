# /usr/bin/python3

import os
import json as JSON
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import Tensor
import torch.utils.data as torchdata
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import CheckpointIO, SingleDevicePlugin
from typing import Any, Dict, Optional, Union
from pandas import DataFrame, Series
from datetime import datetime as Date
from pymongo import MongoClient
from pathlib import Path
from collections import OrderedDict

debugMode = True

# Number of features (this likely changes each year) - 6 teams, stdev and mean
n_features_per_team = 9
n_input_features = n_features_per_team * 6 * 2
# Ratio of input features to hidden layers
hidden_layer_ratios = [1, .5, .3, .3, .3]
# Year of events & matches.
year = 2019
# Whether to do normalizing on data
do_normalize_data = False
do_standardize_data = True

train_losses = []
val_losses = []
val_accuracies = []

trainLossThisEpoch = 0
numTrainData = 1

# Layer 0: 108 neurons
# Layer 2: 54 neurons
# Layer 4: 16 neurons
# Layer 6: 5 neurons
# Layer 8: 2 neuronsZ

num_epochs = 50

def debug(msg):
	if (debugMode == True):
		print(msg)

def main():
	
	init_database()
	
	# getTrainingData('train')
	# createTrainingData(type='train')
	# getDataStandardization()
	train()
	
	print('train_losses = ')
	print(np.array(train_losses).tolist())
	print('val_losses = ')
	print(np.array(val_losses).tolist())
	print('val_accuracies = ')
	print(np.array(val_accuracies).tolist())
	
	plt.plot(range(num_epochs+1), train_losses, label='Training loss')
	plt.plot(range(num_epochs+1), val_losses, label='Validation loss')
	plt.plot(range(num_epochs+1), val_accuracies, label='Prediction accuracy')
	
	plt.xlabel('Epoch (training time)')
	plt.legend()
	plt.show()

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
	
	train_keys = matches.sample(frac=0.7, random_state=376565234)
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
	
	if do_normalize_data:
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
	try:
		dbTrainData.insert_many(modified_items)
	except Exception as e:
		print('Could not insert documents:')
		print(e)
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
		print(len(history.index))
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
			# 'auto_self': 		'$score_self.autoPoints',
			# 'auto_opp': 		'$score_opp.autoPoints',
			# Total points = cargoPoints + panelPoints + autoPoints + habClimbPoints + foulPoints, so not needed
			'total_self':		'$score_self.totalPoints',
			'total_opp':		'$score_opp.totalPoints',
			'rp_self':			'$score_self.rp',
			'rp_opp':			'$score_opp.rp',
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

	global db, dbMatches, dbEvents, dbTrainData, dbOPR, dbStandardize, event_keys
	
	st = Date.now()
	
	db = get_database()
	dbMatches = db['matches']
	dbTrainData = db['trainingdata']
	dbOPR = db['oprs']
	dbStandardize = db['standardizevalues']
	
	event_keys = getEventsByYear(year)

def get_database():

	# Provide the mongodb atlas url to connect python to mongodb using pymongo
	CONNECTION_STRING = "mongodb://localhost:27017/app"
	
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

def standardizeData(df):
	
	records = dbStandardize.find_one({})
	
	if not records:
		print('Could not get standardization factors')
		exit(1)
	
	data = DataFrame.from_records(df['data'])
	
	for key in records:
		record = records[key]
		if not key == '_id':
			mean = record[0]
			std = record[1]
			data[key] = (data[key] - mean) / std # standardize the values
	
	return data
	

def getDataStandardization():
	df = DataFrame(dbTrainData.find({}))
	data = DataFrame.from_records(df['data'])
	
	records = {}
	
	for key in data.columns:
		col = data[key]
		std = col.std()
		mean = col.mean()
		records[key] = [mean, std]
	
	dbStandardize.delete_many({})
	dbStandardize.insert_one(records)
	
	print('Done getting data standardization factors.')

def getTrainingData(traintype):
	
	print('Retrieving training data...')
	
	# Pull from database
	trainFrame = DataFrame(dbTrainData.find({'type': traintype}))
	
	if (len(trainFrame.index) == 0):
		print(f'{traintype} data does not exist')
		exit(1)
		
	data = standardizeData(trainFrame)
	
	# Turn into a Numpy array (removing labels in the process)
	# format: [feature1, ... featureN, blueWon, redWon]
	n_entries = len(data.index)
	
	data_arr = np.empty([n_entries, n_input_features], dtype=float)
	label_arr = np.empty([n_entries], dtype=int)
	
	for i, row in data.iterrows():
		
		features = row.values
		winner = trainFrame['winner'][i]
		
		if (winner == 'blue'):
			label = 0
		elif (winner == 'red'):
			label = 2
		else:
			label = 1
		data_arr[i] = features
		label_arr[i] = label
	
	print(f'Retrieved training data, n_entries={n_entries}')
	
	
	if traintype == 'train':
		global trainLossThisEpoch, numTrainData
		trainLossThisEpoch = 0 # for summing
		numTrainData = n_entries # updated in the training script
		
	
	dataset = MatchDataset()
	dataset.populate(data_arr, label_arr, n_entries)
	
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
	
class CustomCheckpointIO(CheckpointIO):
    def save_checkpoint(
        self, checkpoint: Dict[str, Any], path: Union[str, Path], storage_options: Optional[Any] = None
    ) -> None:
        ...

    def load_checkpoint(self, path: Union[str, Path], storage_options: Optional[Any] = None) -> Dict[str, Any]:
        ...
		
def train():
	# Set fixed random number seed
	pl.seed_everything(42)
	
	lr_monitor = LearningRateMonitor(logging_interval='step')
	# custom_checkpoint_io = CustomCheckpointIO()
	
	mlp = MLP()
	trainer = pl.Trainer(
		gpus=1, 
		max_epochs=num_epochs, 
		check_val_every_n_epoch=1,
		limit_train_batches=1.0, 
		limit_val_batches=1.0, 
		auto_lr_find=True,
		enable_checkpointing=False,
		precision=16,
		# callbacks=[lr_monitor, ModelCheckpoint(save_last=True)],
		callbacks=[lr_monitor],
    	# plugins=[custom_checkpoint_io],
	)
	trainer.tune(mlp)
	trainer.fit(mlp)

class MLP(pl.LightningModule):

	def __init__(self):
		
		activation = nn.ReLU
		
		
		super().__init__()
		
		self.layers = nn.Sequential()
		
		numLastLayer = n_input_features
		
		layers = OrderedDict()
		
		# Dynamically created layers
		for ratio in hidden_layer_ratios:
			numThisLayer = round(numLastLayer * ratio)
			
			print(f'Layer {len(layers)}: {numThisLayer} neurons')
			
			layers[f'layer{len(layers)}'] = nn.Linear(numLastLayer, numThisLayer)
			layers[f'layer{len(layers)}'] = activation()
			numLastLayer = numThisLayer
		layers['output'] = nn.Linear(numLastLayer, 3)
		
		print(layers)
		self.layers = nn.Sequential(layers)
		# self.layers = nn.Sequential(
		# 	nn.Linear(n_input_features, 50),
		# 	activation(),
		# 	nn.Linear(50, 50),
		# 	activation(),
		# 	nn.Linear(50, 3),
		# )
		
		# if numLayer2 > 0:
		# 	self.layers = nn.Sequential(
		# 		nn.Linear(n_input_features, numLayer1),
		# 		activation(),
		# 		nn.Linear(numLayer1, numLayer2),
		# 		activation(),
		# 		nn.Linear(numLayer2, 3),
		# 	)
		# 	print(f'Hidden layer 1: {numLayer1}, Hidden layer 2: {numLayer2}')
		# else:
		# 	self.layers = nn.Sequential(
		# 		nn.Linear(n_input_features, numLayer1),
		# 		activation(),
		# 		nn.Linear(numLayer1, 3),
		# 	)
		# 	print(f'Hidden layer: {numLayer1} neurons')
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
		features, label = batch
		prediction = 	self.layers(features)
		loss = 			self.loss(prediction, label)
		self.log('train_loss', loss)
		global trainLossThisEpoch, numTrainData
		trainLossThisEpoch += loss
		numTrainData += 1
		return loss
	
	def validation_step(self, batch, batch_idx):
		features, label = batch
		prediction = 	self.layers(features)
		loss = 			self.loss(prediction, label)
		self.log('val_loss', loss)
		return loss, prediction, label

	def validation_epoch_end(self, validation_step_outputs):
		sum 		= 0
		i 			= 0
		nCorrect 	= 0
		nWrong 		= 0
		
		for itm in validation_step_outputs:
			loss, prediction, label = itm
			# NN outputs the raw (unnormalized) values; softmax outputs probabilities of each
			probabilities = self.softmax(prediction)
			# Batches of larger than 1 need to iterate
			for j in range(len(probabilities)):
				outcome = label[j]
				guess = probabilities[j]
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
				j += 1
			i += 1
			sum += loss
		
		if len(validation_step_outputs) > 5:
			global trainLossThisEpoch, numTrainData
			val_losses.append(Tensor.cpu(sum / i))
			val_accuracies.append((nCorrect / (nCorrect + nWrong)))
			print(f'trainLossThisEpoch={trainLossThisEpoch}, numTD={numTrainData}')
			train_losses.append(Tensor.cpu(trainLossThisEpoch / numTrainData))
			trainLossThisEpoch = 0 #reset train loss
			numTrainData = 0
		print(f'\n\n\nLoss: {(sum / i):2.2f}, # correct: {nCorrect}, # wrong: {nWrong}, accuracy: {(100 * nCorrect / (nCorrect + nWrong)):2.2f}%')
		print()
	
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters())
		return optimizer

# Torch-based dataset to be wrapped into a DataLoader. This helps reduce memory issues.
class MatchDataset(Dataset):
	
	def __init__(self):
		return
	
	def __len__(self):
		return self.length
	
	def __getitem__(self, index: int):
		return self.features[index], self.labels[index]
	
	def populate(self, features, labels, length):
		self.features = torch.tensor(features, dtype=torch.float32)
		self.labels = torch.tensor(labels, dtype=torch.long)
		self.length = length

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