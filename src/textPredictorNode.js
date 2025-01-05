import {spawn} from "child_process"
export const predictText=(inputText, confidence)=>{
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python3', ['textPredictModel.py', inputText, confidence]);
        let data = '';
        let error = '';
        pythonProcess.stdout.on('data', (chunk) => {
            data += chunk.toString();
        });

        pythonProcess.stderr.on('data', (chunk) => {
            error += chunk.toString();
        });

        pythonProcess.on('close', (code) => {
            if (code === 0) {
                try {
                    const result = JSON.parse(data);
                    resolve(result);
                } catch (parseError) {
                    reject(`Error parsing JSON: ${parseError.message}`);
                }
            } else {
                reject(`Python script exited with code ${code}: ${error}`);
            }
        });
    });
}

