const {spawn} = require('child_process');

// const PYTHON_PATH = "C:\\Users\\draki\\AppData\\Local\\Programs\\Python\\Python39\\python.exe"
const PYTHON_PATH = '/usr/bin/python3';

const NUM_TRAIN = 16000;
const NUM_VAL = 6000;

const NUM_TRAIN_THREADS = 8;
const NUM_VAL_THREADS = 4;

var threads = [];

for (let i = 0; i < NUM_TRAIN_THREADS + NUM_VAL_THREADS; i++) {
	// Divide up the tasks according to the number of threads
	let type, numMin, numMax;
	if (i < NUM_TRAIN_THREADS) {
		type = 'train';
		numMin = Math.floor(NUM_TRAIN * i / NUM_TRAIN_THREADS);
		numMax = Math.ceil(NUM_TRAIN * (i + 1) / NUM_TRAIN_THREADS);
	}
	else {
		type = 'validate';
		numMin = Math.floor(NUM_VAL * (i - NUM_TRAIN_THREADS) / NUM_VAL_THREADS);
		numMax = Math.ceil(NUM_VAL * (i - NUM_TRAIN_THREADS + 1) / NUM_VAL_THREADS);
	}
	
	let thread = spawn(PYTHON_PATH, ['createTrainingData.py', type, numMin, numMax]);
	thread.stdout.on('data', data => {
		let str = data.toString();
		if (!str.includes('Skipping')) {
			console.log(`THREAD ${i}: ${str.replace(/\r.*/g, '')}`);
		}
	});
	thread.stderr.on('data', data => {
		let str = data.toString();
		console.error(`THREAD ${i} ERROR: ${str}`);
	});
	thread.on('exit', code => {
		if (code == 0) {
			console.log(`THREAD ${i} DONE`);
		}
		else {
			console.error(`THREAD ${i} ERROR CODE ${code}`);
		}
	});
}