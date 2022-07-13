#!/usr/bin/python

import scoutradiozmlp as sr
from pandas import DataFrame, Series

def main():
	sr.init_database()
	print(sr.event_keys)
	
	pipeline = [
		{'$match': {
			'event_key': {'$in': sr.event_keys.to_list()}
		}},
		{'$project': {
			'_id': 0,
			'score_breakdown': 0,
			'videos': 0,
			'comp_level': 0,
			'post_result_time': 0,
			'match_number': 0,
			'time': 0,
			'set_number': 0,
		}},
		{'$project': {
			'event_key': 1,
			'key': 1,
			'bluekeys': '$alliances.blue.team_keys',
			'redkeys': '$alliances.red.team_keys',
			'winner': '$winning_alliance',
		}}
	]
	
	matches = DataFrame(sr.dbMatches.aggregate(pipeline))
	OPRs = sr.OPRComparison()
	
	nRight = 0
	nWrong = 0
	nTie = 0
	for i, match in matches.iterrows():
		key = match['key']
		winner = match['winner']
		prediction = OPRs.getPrediction(key)
		outcome = 0 if winner == 'blue' else 2 if winner == 'red' else 0
		if prediction is None:
			pass
		elif prediction == outcome:
			nRight += 1
		else:
			nWrong += 1
		if prediction == 1: nTie += 1
	print(f'Right: {nRight} Wrong: {nWrong}; {100*(nRight / (nRight + nWrong))}%, tie guesses: {nTie}')

if __name__ == '__main__':
	main()