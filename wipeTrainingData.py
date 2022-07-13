#!/usr/bin/python

import scoutradiozmlp as sr

def main():
	sr.init_database()
	sr.wipeTrainingData()

if __name__ == '__main__':
	main()