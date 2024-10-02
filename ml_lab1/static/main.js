const mic_btn = document.querySelector('#mic');
const playback = document.querySelector('.playback');
const text = document.querySelector('#text')
const url = "http://localhost:9001/api/wav_file/analyze"


mic_btn.addEventListener('click', ToggleMic);

let can_record = false;
let is_recording = false;

let recorded = null;

let chunks = [];

function SetupAudio() {
    console.log("setup")
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices
            .getUserMedia({
                audio: true
            })
            .then(SetupStream)
            .catch(err => console.error(err));
    }
}

function SetupStream(stream){
    recorded = new MediaRecorder(stream);
    recorded.ondataavailable = e => {
        chunks.push(e.data);
    }
    recorded.onstop = e => {
        const mimeType = recorded.mimeType;
        const blob = new Blob(chunks, {type: 'audio/mp3'});
        chunks = [];
        const audioUrl = window.URL.createObjectURL(blob);
        playback.src = audioUrl;
        uploadBlob(blob, 'mp3')
    }

    can_record = true;
}
function update_text(json) {
    text.innerText = json.emotion;
}

function uploadBlob(audioBlob, fileType) {
    const formData = new FormData();
    formData.append('file', audioBlob);
    formData.append('type', fileType || 'mp3');
    fetch(url, {method: 'POST', cache: 'no-cache', body: formData}).then((response) => response.json()).then((data) => {update_text(data)}).catch(err => console.error(err));

}
SetupAudio();

function ToggleMic() {
    if (!can_record) return;

    is_recording = !is_recording;

    if (is_recording) {
        recorded.start();
        mic_btn.classList.add("is-recording");
    } else {
        recorded.stop();
        mic_btn.classList.remove("is-recording");
    }
}