import {spawn} from "child_process"
export const predictImage=(imagePath)=>{
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python3', ['imagePredictor.py', imagePath]);
        let data = '';
        pythonProcess.stdout.on('data', (chunk) => {
            data += chunk.toString();
        });
        pythonProcess.stderr.on('data', (chunk) => {
            console.error('Error:', chunk.toString());
        });
        pythonProcess.on('close', (code) => {
            try {
                const result = JSON.parse(data);
                if (result.success) {
                    resolve(result.percentage);
                } else {
                    reject(result.error);
                }
            } catch (error) {
                reject('Failed to parse Python response: ' + error.message);
            }
        });
    });
}

