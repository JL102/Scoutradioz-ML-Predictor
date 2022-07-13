#!/usr/bin/python

import sys
import scoutradiozmlp as sr

def main():
	
	if (len(sys.argv) < 4):
		print('Usage: py (file) [train or validate] startNum endNum')
		exit(1)
	
	type = sys.argv[1]
	start = int(sys.argv[2])
	end = int(sys.argv[3])
	
	if type == 'val': type = 'validate'
	
	if (type != 'train' and type != 'validate'):
		print('Type must be train or validate')
		exit(1)
	
	sr.init_database()
	
	
	# matches, train_keys, validate_keys = sr.getMatchesForTrainingData()
	
	# trainLen 	= len(train_keys.index)
	# valLen 		= len(validate_keys.index)
	
	sr.createTrainingData(type, start, end)

if __name__ == '__main__':
	main()