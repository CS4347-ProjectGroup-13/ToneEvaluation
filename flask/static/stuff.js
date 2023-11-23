class MicrophoneRecorder {
    constructor() {

        this.webAudioRecorder; // our WebAudioRecorder.js recorder yet to be instantiated
        this.currentlyRecording = false; // a boolean to keep track of whether recording is taking place
        this.getUserMediaStream; // our stream from getUserMedia
    }

    async startRecording() {
        if (this.currentlyRecording === false) {
            // the built-in method for capturing audio/video from the user's device
            // pass in the media capture options object and ask for permission to access the microphone
            let options = { 'audio': true, 'video': false };
            navigator.mediaDevices.getUserMedia(options)
            .then(stream => {
        
                this.currentlyRecording = true;
            
              this.getUserMediaStream = stream;

              let AudioContext = window.AudioContext ||  window.webkitAudioContext;
              let audioContext = new AudioContext();
              let source = audioContext.createMediaStreamSource(stream);
              this.webAudioRecorder = new WebAudioRecorder(source, {
                // workerDir: the directory where the WebAudioRecorder.js file lives
                workerDir: 'static/',
                // encoding: type of encoding for our recording ('mp3', 'ogg', or 'wav')
                encoding: 'wav',
                options: {
                  // encodeAfterRecord: our recording won't be usable unless we set this to true
                  encodeAfterRecord: true,
                  // mp3: bitRate: '160 is default, 320 is max quality'
                  mp3: { bitRate: '320' },
                  sampleRate: 16000 // Set the sample rate to 16000    
                }
              });

              this.webAudioRecorder.onComplete = (webAudioRecorder, blob) => {
                this.sendBlob(blob);
              }

              this.webAudioRecorder.onError = (webAudioRecorder, err) => {
                  console.error(err);
              }
              // method that initializes the recording
              this.webAudioRecorder.startRecording();
            }).catch(err => {
                console.error(err);
            });
          } else {
            // if we're already recording, stop the recording and stop the stream from getUserMedia
            console.log('already recording');
          }

    }

    stopRecording() {
        if (this.currentlyRecording === true) {
            let audioTrack = this.getUserMediaStream.getAudioTracks()[0];
            audioTrack.stop();
            this.webAudioRecorder.finishRecording();
            this.currentlyRecording = false;
        }
    }

    async getRecording() {
        while (this.chunks.length === 0) {
            await new Promise(resolve => setTimeout(resolve, 100)); // Wait for 100 milliseconds
        }
        // console.log(this.chunks);
        const blob = new Blob(this.chunks, { type: 'audio/mpeg' }); // Use audio/mpeg MIME type for MP3
        this.sendBlob(blob);
        return blob;
    }

    sendBlob(blob) {
        const currentEndpoint = window.location.origin + '/upload';
        console.log(blob);
        
        // Create a FileReader object
        const reader = new FileReader();

        // Read the blob as Data URL
        reader.readAsDataURL(blob);

        // When the reading is complete
        reader.onloadend = () => {
            // Get the base64 string
            const base64String = reader.result.split(',')[1];

            // Send the base64 string to the current endpoint using a POST request
            fetch(currentEndpoint, {
                method: 'POST',
                body: JSON.stringify({ audio: base64String }),
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => {
                if (response.ok) {
                    console.log('Base64 string sent successfully to', currentEndpoint);
                } else {
                    console.error('Failed to send base64 string to', currentEndpoint);
                }
                return response.json(); // Parse response as JSON
            })
            .then(data => {
                console.log('Response:', data); // Log the response data
            })
            .catch(error => {
                console.error('Error sending base64 string:', error);
            });
        };
    }
}

const recorder = new MicrophoneRecorder();
const RECORDER = recorder;
recorder.startRecording();

recorder.stopRecording();
// const recording = recorder.getRecording();
// console.log(recording);
