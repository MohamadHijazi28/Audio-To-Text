# Audio-To-Text

This project involves the development of an Audio Processing Application using Python. The application provides a graphical user interface (GUI) for users to record audio or select existing audio files, transcribe the audio content, generate summaries, and identify the most repeated word in the transcription. The project utilizes advanced language models like Whisper for audio-to-text transcription and TinyLlama for text summarization and analysis.
The primary goal of this project is to create an interactive tool that simplifies audio processing by integrating several functionalities into one user-friendly application.

# Libraries and Models Used:
•	Tkinter: A Python library for creating graphical user interfaces (GUIs). It is used to build the app’s interface, enabling file selection, audio recording, and text display.

•	Librosa: A Python package used to load and process audio files. It helps in loading audio with a specific sampling rate (16,000 Hz in this case).

•	SoundDevice (sd): Used for recording audio from the system's microphone. The InputStream feature allows continuous audio recording.

•	Scipy: Specifically, the write method from scipy.io.wavfile is used to save recorded audio as .wav files.

•	Transformers (Hugging Face):

  o	Whisper Model: A pre-trained model from OpenAI, used for speech-to-text transcription.

  o	TinyLlama Model: A smaller language model employed for text summarization and determining the most repeated word.

•	Numpy: A numerical processing library used for handling audio data and performing concatenation operations.

# Application Features:
•	Audio Selection: The app provides an option for the user to select an audio file (wav format) from their system.

•	Audio Recording: Users can record audio through their microphone, and the app saves it as a .wav file.

•	Transcription: After an audio file is selected or recorded, the Whisper model is used to generate the transcription of the audio.

•	Text Summarization: Once the audio is transcribed, the TinyLlama model can generate a summary of the text.

•	Most Repeated Word Detection: The TinyLlama model is also used to find the most frequently repeated word in the transcription.

# Challenges and Solutions:
1.	Long Audio Processing: To handle longer audio files, the transcription function splits the audio into smaller chunks.
2.	Model Integration: Pre-trained models (Whisper and TinyLlama) are integrated seamlessly into the application.
3.	Real-time Audio Recording: Implemented using a callback function with sounddevice to capture audio in real-time.

# Future Improvements:
•	Support for Additional Formats: Adding support for other audio file formats, such as MP3, to enhance flexibility.

•	Real-time Transcription: Implementing real-time transcription for recorded audio without saving the file first.

•	Performance Optimization: Reducing the processing time for large audio files

# Conclusion:
The Audio Processing App demonstrates the effective integration of various technologies to create a useful tool for audio analysis. It showcases the power of combining speech recognition, natural language processing, and a user-friendly interface to extract valuable information from audio content.
