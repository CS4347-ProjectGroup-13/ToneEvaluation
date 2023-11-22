class MicrophoneRecorder {
    constructor() {
        this.mediaStream = null;
        this.mediaRecorder = null;
        this.chunks = [];
        this.isRecording = false;
    }

    async startRecording() {
        if (this.isRecording) {
            console.log('Microphone recording is already started.');
            return;
        }

        try {
            this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(this.mediaStream);

            this.mediaRecorder.addEventListener('dataavailable', (event) => {
                this.chunks.push(event.data);
                // console.log(this.chunks);
            });

            this.mediaRecorder.start();
            this.isRecording = true;
            this.chunks = []; // Clean the chunks array when recording starts
        } catch (error) {
            console.error('Error starting microphone recording:', error);
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
            this.mediaStream.getTracks().forEach((track) => track.stop());
            this.isRecording = false;
        }
    }

    async getRecording() {
        while (this.chunks.length === 0) {
            await new Promise(resolve => setTimeout(resolve, 100)); // Wait for 100 milliseconds
        }
        // console.log(this.chunks);
        // const blob = new Blob(this.chunks, { type: 'audio/webm' });
        this.sendBlob(this.chunks[0]); 
        return blob;
    }

    sendBlob(blob) {
        const currentEndpoint = '/upload'; 
        console.log(blob);
        // Create a FormData object
        const formData = new FormData();
        formData.append('audio', blob, 'recording.webm');

        // Send the FormData object to the current endpoint using a POST request
        fetch(currentEndpoint, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (response.ok) {
                console.log('Blob sent successfully to', currentEndpoint);
            } else {
                console.error('Failed to send blob to', currentEndpoint);
            }
        })
        .catch(error => {
            console.error('Error sending blob:', error);
        });
    }

    
}

const recorder = new MicrophoneRecorder();
const RECORDER = recorder
recorder.startRecording();


recorder.stopRecording();
// const recording = recorder.getRecording();
// console.log(recording);

