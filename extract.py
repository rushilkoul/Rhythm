import librosa as lb
import numpy as np
import os

# song ki place
song = "song.mp3"


# load song
y, sr = lb.load(song, sr=None, mono=True)


# split songs into segments each of 5 seconds
segments_size = 5 * sr
segments = []
for i in range(0, y.size, segments_size):
    segment = y[i : i + segments_size]
    if len(segment) < sr:  # to remove any very small ass segments
        continue
    else:
        segments.append(segment)


# feature extraction

# A: Calm / Intro / Acoustic (low energy)
# B: Melodic Verse (medium energy)
# C: Groove / Chorus (high energy)
# D: Peak / Climax / Drop (very high energy)

def extract_features(segment, sr):
    features = {}

    rms = lb.feature.rms(y=segment)  # how loud the music is

    features["energy"] = float(np.mean(rms))

    # zero crossing rate how freq does the audio amplitude crosses 0
    zcr = lb.feature.zero_crossing_rate(y=segment)

    features["zcr"] = float(np.mean(zcr))

    # spectral centroid : basically the center of gravity in the frequency

    centroid = lb.feature.spectral_centroid(y=segment, sr=sr)

    features["centroid"] = float(np.mean(centroid))

    # spacial bandwidth : how spread out the feq are

    bandwidth = lb.feature.spectral_bandwidth(y=segment, sr=sr)

    features["bandwidth"] = float(np.mean(bandwidth))

    # SPECTRAL ROLLOFF : idea of how much high frequency content i present

    rolloff = lb.feature.spectral_rolloff(y=segment, sr=sr)

    features["rolloff"] = float(np.mean(rolloff))

    # CHROMA : pitch of the music , 12 pitches only , octaves are ignored

    chroma = lb.feature.chroma_stft(y=segment, sr=sr)

    features["chroma"] = np.mean(
        chroma, axis=1  # 12 pitches ke liye ek mean value chaiye
    )

    # MFCC : overall shape of sound that follows humans prespective , used for speech and music recog

    mfcc = lb.feature.mfcc(y=segment, sr=sr, n_mfcc=13)  # 13 diff values

    features["mfcc"] = np.mean(mfcc, axis=1)  # each 13 values one mean value

    # TEMPO : bpm of the music

    tempo, _ = lb.beat.beat_track(y=segment, sr=sr)

    features["tempo"] = float(np.asarray(tempo).flatten()[0])

    return features


# loop for each segment :


all_features = []

for i, segment in enumerate(segments):

    features = extract_features(segment, sr)

    all_features.append(features)

    print(f"\nSegment {i + 1}")

    print("Energy:", features["energy"])

    print("ZCR:", features["zcr"])

    print("Centroid:", features["centroid"])

    print("Bandwidth:", features["bandwidth"])

    print("Rolloff:", features["rolloff"])

    print("Tempo:", features["tempo"])

    print("Chroma:", features["chroma"])

    print("MFCC:", features["mfcc"])
