import {spawn} from "child_process"
export const predictHR=(edaValue, hrValue)=>{
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python3', ['hrPredictor.py', edaValue, hrValue]);

        let data = '';
        let error = '';

        pythonProcess.stdout.on('data', (chunk) => {
            data = chunk.toString();
        });

        pythonProcess.stderr.on('data', (chunk) => {
            error = chunk.toString();
        });

        pythonProcess.on('close', (code) => {
            if (code === 0) {
                try {
                    const prediction = JSON.parse(data.trim()).probability;
                    resolve(prediction);
                } catch (parseError) {
                    reject(`Error parsing prediction: ${parseError.message}`);
                }
            } else {
                reject(`Python script exited with code ${code}: ${error}`);
            }
        });
    });
}

