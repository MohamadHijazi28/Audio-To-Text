import tkinter as tk
from tkinter import filedialog, messagebox
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import WhisperForConditionalGeneration, WhisperProcessor, \
    AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

# Load the Whisper model and processor
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")

# Load the TinyLlama model and tokenizer
llama_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
llama_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Global variables for recording
is_recording = False
audio_data = []
fs = 44100  # Sample rate for recording
selected_file_path = None
recording_stream = None  # Stream object


# Function to process text with TinyLlama
def process_with_llama(text, task):
    prompt = f"{task}:\n\n{text}\n\nResult:"
    inputs = llama_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        output = llama_model.generate(**inputs, max_new_tokens=150, num_return_sequences=1)
    result = llama_tokenizer.decode(output[0], skip_special_tokens=True)
    return result.split("Result:")[-1].strip()


# Define functions for summarization and getting the most repeated word
def summarize_text(text):
    return process_with_llama(text, "Summarize the following text")


def get_most_repeated_word(text):
    return process_with_llama(text,
                              "What is the most repeated meaningful word in the following text? Provide only one word, no explanation")


def split_audio(audio, sr, chunk_length=30):
    chunk_size = sr * chunk_length
    return [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]


def transcribe_audio(audio_file_path):
    audio, sr = librosa.load(audio_file_path, sr=16000)

    # Split audio into chunks
    audio_chunks = split_audio(audio, sr)

    transcriptions = []
    for chunk in audio_chunks:
        input_features = whisper_processor(chunk, sampling_rate=sr, return_tensors="pt").input_features
        generated_ids = whisper_model.generate(input_features)
        transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        transcriptions.append(transcription)

    # Combine all transcriptions
    full_transcription = " ".join(transcriptions)
    return full_transcription


def record_audio_callback(indata, frames, time, status):
    global audio_data
    audio_data.append(indata.copy())


def toggle_recording():
    global is_recording, audio_data, recording_stream, selected_file_path
    if is_recording:
        # Stop recording
        if recording_stream is not None:
            recording_stream.stop()
            recording_stream.close()

        audio_data = np.concatenate(audio_data, axis=0)
        write('output.wav', fs, audio_data)  # Save the recorded audio to a file
        file_path_label.config(text="Recorded file saved: output.wav")
        record_button.config(text="Record Audio")
        is_recording = False
        audio_data = []  # Reset after saving

        selected_file_path = 'output.wav'
    else:
        # Start recording
        audio_data = []
        recording_stream = sd.InputStream(callback=record_audio_callback, channels=1, samplerate=fs)
        recording_stream.start()  # Start the stream
        record_button.config(text="Stop Recording")
        file_path_label.config(text="Recording...")
        is_recording = True


# GUI Functions
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        file_path_label.config(text=f"Selected file: {file_path}")
        global selected_file_path
        selected_file_path = file_path


def generate_transcription():
    if not selected_file_path:
        messagebox.showerror("Error", "No audio file selected or recorded")
        return
    transcription = transcribe_audio(selected_file_path)
    transcription_text.delete(1.0, tk.END)  # Clear the text area before inserting new text
    transcription_text.insert(tk.END, transcription)


def generate_summary():
    transcription = transcription_text.get(1.0, tk.END).strip()
    if not transcription:
        messagebox.showerror("Error", "Please generate a transcription first")
        return
    summary = summarize_text(transcription)
    summary_text.delete(1.0, tk.END)  # Clear the text area before inserting new text
    summary_text.insert(tk.END, summary)


def generate_most_repeated_word():
    transcription = transcription_text.get(1.0, tk.END).strip()
    if not transcription:
        messagebox.showerror("Error", "Please generate a transcription first")
        return
    most_repeated = get_most_repeated_word(transcription)
    most_repeated_text.delete(1.0, tk.END)  # Clear the text area before inserting new text
    most_repeated_text.insert(tk.END, most_repeated)


# GUI setup
root = tk.Tk()
root.title("Audio Processing App")
root.geometry("700x700")  # Increase the size of the window
root.config(bg="lightblue")

# Select or Record Audio
button_frame = tk.Frame(root, bg="lightblue")  # Frame to hold the buttons on the same line
button_frame.pack(pady=10)

select_file_button = tk.Button(button_frame, text="Select Audio File", command=select_file, width=20,
                               font=("Helvetica", 11), bg="coral")
select_file_button.grid(row=0, column=0, padx=10)

record_button = tk.Button(button_frame, text="Record Audio", command=toggle_recording, width=20, font=("Helvetica", 11),
                          bg="teal")
record_button.grid(row=0, column=1, padx=10)

file_path_label = tk.Label(root, text="No file selected or recorded", bg="lightblue")
file_path_label.pack(pady=5)

# Text Output Areas for Transcription, Summarization, Most Repeated Word
transcription_label = tk.Label(root, text="Transcription", font=("Helvetica", 11), bg="gray")
transcription_label.pack(pady=5)
transcription_text = tk.Text(root, height=10, width=80, font=12)
transcription_text.pack(pady=5)

# Button for generating transcription
transcription_button = tk.Button(root, text="Generate Transcription", command=generate_transcription, width=25,
                                 font=("Helvetica", 11), bg="teal")
transcription_button.pack(pady=5)

summary_label = tk.Label(root, text="Summary", font=("Helvetica", 11), bg="gray")
summary_label.pack(pady=5)
summary_text = tk.Text(root, height=5, width=80, font=12)
summary_text.pack(pady=5)

# Button for generating summary
summary_button = tk.Button(root, text="Generate Summary", command=generate_summary, width=25, font=("Helvetica", 11),
                           bg="teal")
summary_button.pack(pady=5)

most_repeated_label = tk.Label(root, text="Most Repeated Word", font=("Helvetica", 11), bg="gray")
most_repeated_label.pack(pady=5)
most_repeated_text = tk.Text(root, height=2, width=80, font=12)
most_repeated_text.pack(pady=5)

# Button for generating most repeated word
most_repeated_button = tk.Button(root, text="Generate Most Repeated Word", command=generate_most_repeated_word,
                                 width=25, font=("Helvetica", 11), bg="teal")
most_repeated_button.pack(pady=5)

# Start the GUI loop
root.mainloop()
