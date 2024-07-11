import speech_recognition as sr
from speechbrain.inference import SpeakerRecognition
import soundfile as sf
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json
import requests
import time

# FastAPI url
url = "http://127.0.0.1:8000/chat"

# Initialize recognizer
recognizer = sr.Recognizer()

# Load the pre-trained speaker recognition model
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp")

# Load Vosk model
model_path = "models/vosk-model-small-en-us-0.15"

try:
    vosk_model = Model(model_path)
    print("Vosk model loaded successfully.")
except Exception as e:
    print(f"Failed to create a model: {e}")
    exit(1)

# Recording an audio file for the known speaker
def set_speaker(filename="known_speaker_sample.wav", duration=10, fs=44100):
    print(f"Recording a sample for the known speaker ...")
    print(f"Please speak for {duration} seconds")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    sf.write(filename, recording, fs)  # Save the recording as a WAV file
    print(f"Audio saved as {filename}")
    return filename

def load_known_speaker_sample():
    filename="known_speaker_sample.wav"
    return filename
    
# Check if it is the known speaker using speechbrain
def verify_speaker(audio_file1, audio_file2):
    score, prediction = verification.verify_files(audio_file1, audio_file2)
    return score.item(), prediction.item()

# Function to listen for the keyword "Hey"
def listen_for_keyword():
    recognizer = sr.Recognizer()
    keyword_detected = False
    while not keyword_detected:
        try:
            with sr.Microphone() as mic:
                print("Listening for keyword...")
                recognizer.adjust_for_ambient_noise(mic, duration=1)
                audio = recognizer.listen(mic)
                
                # Use Vosk for offline keyword detection
                kaldi_recognizer = KaldiRecognizer(vosk_model, mic.SAMPLE_RATE)
                kaldi_recognizer.AcceptWaveform(audio.get_wav_data())
                result = json.loads(kaldi_recognizer.FinalResult())
                text = result.get('text', '')
                print(f"Detected: {text}")
                if "hey" in text:
                    print("Keyword detected.")
                    keyword_detected = True
        except sr.RequestError as e:
            print(f"An error occurred with the recognizer: {e}")

# Main function that allows speech to text after calling the speaker identification function
def listen():
    set_new_speaker = input("Do you want to set a new speaker? (yes/no): ").strip().lower()
    
    if set_new_speaker == 'yes':
        known_speaker_sample = set_speaker()
    else:
        # Load the existing known speaker sample
        known_speaker_sample = load_known_speaker_sample()
    
    print("Starting the speech and speaker recognition program...")

    recognizer = sr.Recognizer()
    
    while True:
        try:
            # Listen for the keyword "Hey"
            listen_for_keyword()
            
            # Adding a short pause to allow the system to reset
            time.sleep(0.5)
            
            with sr.Microphone() as mic:
                print("Listening...")
                recognizer.adjust_for_ambient_noise(mic, duration=1)
                time.sleep(0.5)  # Give additional time for adjustment
                audio = recognizer.listen(mic)
                print("Processing speaker identification...")

                # Save the recorded audio to a file for speaker verification
                with open("temp_audio.wav", "wb") as f:
                    f.write(audio.get_wav_data())

                # Predict if the speaker is the known speaker
                score, prediction = verify_speaker("temp_audio.wav", known_speaker_sample)
                if prediction:
                    print(f"Speaker identified as known speaker with score: {score}")
                    
                    # Use Vosk for offline speech recognition
                    kaldi_recognizer = KaldiRecognizer(vosk_model, mic.SAMPLE_RATE)
                    kaldi_recognizer.AcceptWaveform(audio.get_wav_data())
                    result = json.loads(kaldi_recognizer.FinalResult())
                    text = result.get('text', '')
                    if text:
                        text = text.lower()
                        print(f"Recognized: {text}")
                        print(f"Generating answer")
                        start_time = time.time()
                        payload = {"user_input": text}
                        headers = {"Content-Type": "application/json"}
                        response = requests.post(url, json=payload, headers=headers)
                        end_time = time.time()
                        print(response.json())
                        print(f"Time taken: {end_time - start_time}")
                    else:
                        print("Could not understand the audio, please try again.")
                else:
                    print(f"Speaker not recognized. Score: {score}. Please try again.")
        except sr.UnknownValueError:
            print("Could not understand the audio, please try again.")
            recognizer = sr.Recognizer()
            continue
        except sr.RequestError as e:
            print(f"Could not request results from the recognizer service; {e}")
        except AttributeError as e:
            print(f"An error occurred: {e}. This usually means the microphone failed to initialize.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

    print("Program ended.")

if __name__ == "__main__":
    listen()
