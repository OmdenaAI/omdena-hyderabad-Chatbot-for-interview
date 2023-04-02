# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 21:00:04 2023

@author: Aren Wilson Wright, Ponniah Kameswaran
"""

import os
import re
import sox
import spacy
import crepe
import torch
import shutil
import librosa
import numpy as np
import pandas as pd
import datetime as dt
from pytube import YouTube
from pydub import AudioSegment
from numpy.linalg import norm
from scipy.special import expit
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline, AutoModelForSequenceClassification, AutoTokenizer
from pyannote.audio import Inference, Model, Pipeline

def make_directories(directories):
    """
    This function creates subdirectories from a user-provided list
    """
    ROOT = os.getcwd()
    for directory in directories:
        os.makedirs(os.path.join(ROOT, directory), exist_ok=True)


def retrieve_audio(video_link, output_dir):
    """
    This function scrapes the audio from a YouTube video and saves it in the output directory
    """
    ROOT = os.getcwd()
    output = os.path.join(ROOT, output_dir)
    os.makedirs(output, exist_ok=True)
    try:
        video = YouTube(video_link)
        audio = video.streams.filter(only_audio=True, file_extension='mp4').first()
        audio.download(output_dir)
    except:
        print("Connection error")


def convert_to_wav(input_path, output_dir):
    """
    This function converts and audio file into the .wav format and saves it in the output directory
    """
    ROOT = os.getcwd()
    output = os.path.join(ROOT, output_dir)
    os.makedirs(output, exist_ok=True)

    file_name = re.split("/|\.", input_path)[-2]

    audio = AudioSegment.from_file(input_path)
    audio.export(f"{output_dir}/{file_name}.wav", format='wav')


def diarization_profiler(file, diarization_pipeline):
    """
    This function diarizes the provided audio file and calculates some summary statistics about each turn, which are stored in the dataframes c_turns and turn_profile
    """
    diarization = diarization_pipeline(file['audio'])

    turns = pd.DataFrame(columns=['speaker', 'start_time', 'end_time'])

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns = turns.append({'speaker' : speaker, 'start_time' : turn.start, 'end_time' : turn.end}, ignore_index=True)

    speakers = {speaker : i for i, speaker in enumerate(sorted(turns.speaker.unique()))}

    turns['speaker'].replace(speakers.keys(), speakers.values(), inplace=True)

    ignore_silence_duration = 2.0
    turns.sort_values(by=['start_time'], ascending=True, inplace=True)
    n_speakers = len(turns.speaker.unique())

    c_turns = pd.DataFrame(columns=["speaker", "start_time", "end_time", "turn_type", "gap"])
    last_speech = np.empty((n_speakers, 3))
    last_speech[:, :] = -1
    insert_count = 0
    turn_type = ""
    mutual_silence = 0.0

    for index, row in turns.iterrows():
        gap = 0.0
        other_speakers = [i for i in range(n_speakers) if i != int(row.speaker)]  

        if (((~(last_speech[other_speakers, 1] > row.start_time)).all()) and 
            ((row.start_time - last_speech[int(row.speaker), 1]) < ignore_silence_duration) and 
            (last_speech[int(row.speaker), 1] > 0)
            ): 
                c_turns.loc[int(last_speech[int(row.speaker), 2]), "end_time"] = row.end_time
                last_speech[int(row.speaker), 1] = row.end_time
        else:
            if insert_count == 0:
                turn_type = "LAUNCH"
            elif np.sum([True for x in last_speech[other_speakers] if (row.start_time < x[1] and row.end_time >= x[1]) ]) > 0:
                turn_type = "INTERRUPTION"
            elif np.sum([True for x in last_speech[other_speakers] if (row.start_time < x[1] and row.end_time <= x[1]) ]) > 0:
                turn_type = "OVERLAP"
            elif np.argmax(last_speech[:, 1]) == row.speaker:
                turn_type = "CONTINUE"
                gap = row.start_time - np.max(last_speech[:, 1])
                mutual_silence += row.start_time - np.max(last_speech[:, 1])
            else:
                turn_type = "RESPONSE"
                gap = row.start_time - np.max(last_speech[:, 1])
                if row.start_time - np.max(last_speech[:, 1]) > ignore_silence_duration:
                    mutual_silence += row.start_time - np.max(last_speech[:, 1])
                
            c_turns = c_turns.append({"speaker": int(row.speaker), 
                                      "start_time": row.start_time, 
                                      "end_time": row.end_time, 
                                      "turn_type": turn_type,
                                      "gap" : gap}, 
                                     ignore_index=True)
            last_speech[int(row.speaker)] = [row.start_time, row.end_time, insert_count]
            insert_count += 1

    c_turns["duration"] = c_turns["end_time"] - c_turns["start_time"]

    turn_profile = pd.DataFrame(columns=["speaker"])

    if n_speakers == 1:
        audio_type = "SOLO"
        mutual_silence = np.nan
    else:
        audio_type = "GROUP"

    turn_profile["speaker"] = [i for i in range(n_speakers)]
    turn_profile["audio_type"] = audio_type
    turn_profile["total_turn_duration"] = c_turns.groupby(by="speaker").duration.sum()
    turn_profile["turn_duration"] = [c_turns[(c_turns.speaker==x)].duration.describe().to_dict() for x in turn_profile.index]
    turn_profile["speaking_percent"] = np.divide(turn_profile.total_turn_duration, c_turns.duration.sum())
    turn_profile["mutual_silence"] = mutual_silence
    turn_profile["response_time"] = [c_turns[(c_turns.speaker==x) & (c_turns.turn_type=="RESPONSE")].gap.describe().to_dict() for x in turn_profile.index]
    turn_profile["interruptions"] = [c_turns[(c_turns.speaker==x) & (c_turns.turn_type=="INTERRUPTION")].duration.describe().to_dict() for x in turn_profile.index]
    turn_profile["overlap_duration"] = [c_turns[(c_turns.speaker==x) & (c_turns.turn_type=="OVERLAP")].duration.sum() for x in turn_profile.index]
    turn_profile["overlap"] = [c_turns[(c_turns.speaker==x) & (c_turns.turn_type=="OVERLAP")].duration.describe().to_dict() for x in turn_profile.index]

    return c_turns, turn_profile


def get_turns(file, c_turns, asr_processor, asr_model, offset=0, samplerate=16000):
    """
    This function consolidates consecutive turns with the same speaker into a single turn on the basis of c_turns and generates the corresponding audio files and transcriptions
    """
    ROOT = os.getcwd()
    tmp_dir = os.path.join(ROOT, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    asr_model.eval()

    for index, row in c_turns.iterrows():
        tfm = sox.Transformer()
        tfm.trim(row.start_time - offset, row.end_time + offset)
        tfm.build_file(f"input/{file['name']}.wav", f"tmp/{file['name']}_turn_{index}.wav")
        audio, sr = librosa.load(f"tmp/{file['name']}_turn_{index}.wav", sr=samplerate)

    c_turns.loc[index, "audio_file"] = f"tmp/{file['name']}_turn_{index}.wav"

    with torch.no_grad():
        input_values = asr_processor(audio, return_tensors="pt", padding="longest", sampling_rate=samplerate).input_values
        logits = asr_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        pred_transcript = asr_processor.batch_decode(predicted_ids)

        #torch.cuda.empty_cache() --clear memory if using GPU acceleration
        #spellcheck goes here if necessary  

        c_turns.loc[index, "transcript"] = pred_transcript[0].lower()

    return c_turns


def get_speaker_data(c_turns, turn_profile):
    """
    This function extracts speaker-level audio and text data from the list of turns and appends it to the turn_profile DataFrame
    """
    for index, row in turn_profile.iterrows():
        turn_profile.loc[index, "transcript"] = " ".join(c_turns[c_turns.speaker == index]['transcript'].to_list())
        audio_files = c_turns[c_turns.speaker == index]['audio_file'].to_list()

        if len(audio_files) > 1:
            cbn =  sox.Combiner()
            cbn.build(audio_files, f"tmp/speaker_{index}.wav", "concatenate")
        else:
            tfm = sox.Transformer()
            tfm.build_file(audio_files[0], f"tmp/speaker_{index}.wav")

        turn_profile.loc[index, "audio_file"] =  f"tmp/speaker_{index}.wav"

    return turn_profile


def get_emotions(turn_profile, sentiment_pipeline):
    """
    This function assigns a sentiment label to a text using the provided model
    """
    for index, row in turn_profile.iterrows():
        turn_profile.loc[index, 'emotions'] =  sentiment_pipeline(row.transcript[:512])[0]['label']

    return turn_profile


def get_turn_topics(c_turns, turn_profile, topic_tokenizer, topic_model):
    """
    This function assigns topic labels to each turn in c_turns and then appends the number of abrupt topic changes per speaker to turn_profile 
    """
    class_mapping = topic_model.config.id2label

    for index, row in c_turns.iterrows():
        tokens = topic_tokenizer(row.transcript[:514], return_tensors='pt')

        output = topic_model(**tokens)

        scores = output[0][0].detach().numpy()
        scores = expit(scores)
        predictions = (scores >= 0.5) * 1

        topics = []

        for i in range(len(predictions)):
            if predictions[i]:
                topics.append(class_mapping[i])

        c_turns.loc[index, 'topics'] = ",".join(topics)

    responses = c_turns[(c_turns['turn_type'] == 'RESPONSE') | (c_turns['turn_type'] == 'LAUNCH')].reset_index(drop=True)
    turn_profile['topic_shifts'] = 0

    for index, row in responses.iterrows():
        if index == 0:
            pass
        else:
            if set(responses.loc[index - 1, 'topics'].split(",")).intersection(set(responses.loc[index, 'topics'].split(","))) == set(): 
                turn_profile.loc[row.speaker, 'topic_shifts'] += 1

    return c_turns, turn_profile


def cosine_similarity(a, b):
    """
    This function calculates the cosine similarity between two vectors.
    """
    cos_sim = (a @ b.T) / (norm(a)*norm(b))
    return cos_sim


def calculate_distance(sample_audio, turn_profile, inference_model):
    """
    This function calculates the cosine similarity between the user's sample_audio and each unique speaker in the conversation using the provided model and appends this data to turn_profile
    """
    sample_audio = inference_model(sample_audio)

    for index, row in turn_profile.iterrows():
        turn_profile.loc[index, 'similarity'] =  cosine_similarity(sample_audio, inference_model(row.audio_file))

    return turn_profile


def get_user_data(turn_profile, threshold):
    """
    This function identifies the speaker from the sample_audio file with one of the speakers in the turn_profile DataFrame within a threshold"
    """
    if turn_profile['similarity'].max() < threshold:
        print("User could not be identified. Please manually identify the user.") #way to log
    else:
        speaker_data = turn_profile[turn_profile['similarity'] == turn_profile['similarity'].max()]

    return speaker_data


def extract_audio_features(audio_file, samplerate):
    """
    This function calculates descriptive statistics for the spectral flatness, volume, and pitch of the provided audio file
    """
    audio, sr = librosa.load(audio_file, sr=samplerate)

    flatness = pd.DataFrame(librosa.feature.spectral_flatness(y=audio).T).describe().T
    loudness = pd.DataFrame(librosa.feature.rms(audio).T).describe().T
    time, frequency, confidence, activation = crepe.predict(audio, sr)
    frequency = pd.DataFrame(frequency.T).describe().T

    return flatness, loudness, frequency


def get_filler_word_percent(text, lang_model):
    """
    This function calculates the percentage of filler words in a text using the provided language model
    """
    doc = lang_model(text)
    filler_words = [token.text for token in doc if token.pos_ == 'INTJ']
    filler_word_pr =  len(filler_words) / len(doc)

    return filler_word_pr


def get_features(file, sample_audio, diarization_pipeline, asr_processor, asr_model, sentiment_pipeline, topic_tokenizer, topic_model, inference_model, lang_model, offset=0, samplerate=16000, display=False, threshold=.5):
    """
    This function identifies the user in the provided audio file, extracts their audio, and calculates and returns various features of their speech
    """
    # denoising goes here

    c_turns, turn_profile  = diarization_profiler(file, diarization_pipeline)

    c_turns = get_turns(file, c_turns, asr_processor, asr_model, offset, samplerate)

    # spellcheck goes here

    turn_profile = get_speaker_data(c_turns, turn_profile)

    turn_profile = get_emotions(turn_profile, sentiment_pipeline)

    c_turns, turn_profile = get_turn_topics(c_turns, turn_profile, topic_tokenizer, topic_model)

    turn_profile = calculate_distance(sample_audio, turn_profile, inference_model)

    speaker_data = get_user_data(turn_profile, threshold).reset_index(drop=True)

    flatness, loudness, frequency = extract_audio_features(speaker_data.loc[0, 'audio_file'], samplerate)

    features = {}

    features['date'] = dt.datetime.now()

    features['mutual_silence'] =  speaker_data.loc[0, 'mutual_silence']

    features['overlap_duration'] = speaker_data.loc[0, 'overlap_duration']

    features['interruptions'] = speaker_data.loc[0, 'interruptions']['count'] 

    features['total_turn_duration'] = speaker_data.loc[0, 'total_turn_duration']

    features['speaking_percent'] = speaker_data.loc[0, 'speaking_percent']

    features['response_time'] = speaker_data.loc[0, 'response_time']['mean']

    features['topic_shifts'] = speaker_data.loc[0, 'topic_shifts']

    features['emotions'] = speaker_data.loc[0, 'emotions']

    features['words_per_minute'] = len(speaker_data.loc[0, 'transcript'].split(" ")) / (speaker_data.loc[0, 'total_turn_duration'] / 60) 

    features['fillerword_percent'] = get_filler_word_percent(speaker_data.loc[0, 'transcript'], lang_model)

    features['mean_spectral_flatness'] = flatness.loc[0, 'mean'] 

    features['spectral_flatness_std'] = flatness.loc[0, 'std'] 

    features['mean_pitch'] = frequency.loc[0, 'mean'] 

    features['pitch_std'] = frequency.loc[0, 'std'] 

    features['mean_volume'] = loudness.loc[0, 'mean'] 

    features['volume_std'] = loudness.loc[0, 'std'] 

    if speaker_data.loc[0, 'audio_type'] == 'GROUP':
        features['is_group'] = 1
    else:
        features['is_group'] = 0

    for file in os.scandir('tmp'):
        os.remove(file.path)

    return features


if __name__ == "__main__":
    #set up our folder structure
    make_directories(['input', 'tmp', 'user_data'])

    # setup hardcoded links for audio files
    speaker_ref_video_link = "https://www.youtube.com/watch?v=6ObqydfPGLI"
    speaker_ref_mp4_path = "tmp/Yale Professor Tony Leiserowitz Discusses American Perceptions of Climate Change.mp4"
    speaker_ref_wav_path = "tmp/Yale Professor Tony Leiserowitz Discusses American Perceptions of Climate Change.wav"
    speaker_ref_file_path = "user_data/speaker_sample.wav"
    
    meeting_rec_video_link = "https://www.youtube.com/watch?v=T-JVKqpvt2c&t=1996s"
    meeting_rec_mp4_path = "tmp/Dr Anthony Leiserowitzs Keynote Address - 2017 Conference.mp4"
    meeting_rec_wav_path = "tmp/Dr Anthony Leiserowitzs Keynote Address - 2017 Conference.wav"
    meeting_rec_file_path = "tmp/Dr_Anthony_Leiserowitzs_Keynote Address_2017_Conference_trimmed.wav"

    #create a speaker sample that will be used for speaker identification purpose
    retrieve_audio(speaker_ref_video_link, "tmp")
    convert_to_wav(speaker_ref_mp4_path, "tmp")
    tfm = sox.Transformer()
    tfm.trim(10, 40)
    tfm.build_file(speaker_ref_wav_path, speaker_ref_file_path)
    
    #retrieve meeting recording    
    retrieve_audio(meeting_rec_video_link, "tmp")
    convert_to_wav(meeting_rec_mp4_path, "tmp")
    tfm = sox.Transformer()
    tfm.trim(75, 105)
    tfm.build_file(meeting_rec_wav_path, meeting_rec_file_path)
    
    ROOT = os.getcwd()
    shutil.copyfile(meeting_rec_file_path, os.path.join(ROOT, 'input/chunk.wav'))
    
    #download the required models; to use the pyannote modules go to https://github.com/pyannote/pyannote-audio and follow the instructions
    auth_token = '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' #your huggingface authorization token here
    
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=auth_token)
    asr_processor = Wav2Vec2Processor.from_pretrained("gngpostalsrvc/w2v2-ami")
    asr_model = Wav2Vec2ForCTC.from_pretrained("gngpostalsrvc/w2v2-ami")
    inference_model = Inference(Model.from_pretrained("pyannote/embedding", use_auth_token=auth_token), window="whole")
    sentiment_pipeline = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
    lang_model = spacy.load("en_core_web_sm")
    topic_tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/tweet-topic-21-multi')
    topic_model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/tweet-topic-21-multi')
    
    # initialized features data frame with column titles
    output = pd.DataFrame(columns=['date', 'mutual_silence', 'overlap_duration', 'interruptions', 'total_turn_duration', 'speaking_percent', 'response_time', 'emotions', 'topic_shifts', 'words_per_minute', 'fillerword_percent', 'mean_spectral_flatness', 'spectral_flatness_std', 'mean_pitch', 'pitch_std', 'mean_volume', 'volume_std', 'is_group', 'skill', 'score'])
    
    #set the random seed to ensure reproducibility
    np.random.seed(42) 
    
    # process the audio file and extract soft skill features
    for chunk in os.listdir('input'):
          chunk = {'name' : chunk.split(".")[0], 'audio' : f"input/{chunk}"}
        
          features = get_features(chunk, speaker_ref_file_path, diarization_pipeline, asr_processor, asr_model, sentiment_pipeline, topic_tokenizer, topic_model, inference_model, lang_model, threshold=.1)
        
          features['skill'] = 'communication'
          features['score'] = np.random.randint(6) 
          output = output.append(features, ignore_index=True)
        
          break
    
    print(output)

