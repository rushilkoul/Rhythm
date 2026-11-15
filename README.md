## Rhythm: music genre classification

A small experiment in using TOC concepts to classify music.

Instead of using an ML model, Rhythm takes a song, converts it into a sequence of discrete symbols based on its energy levels, and runs that sequence through a set of genre-specific NFAs.

## How it works

The pipeline is simple:

### 1. Audio Processing

- Takes an audio file (`.mp3`) as input.
- Splits it into 5-second segments.
- Uses Librosa to calculate the RMS energy of each segment.

### 2. Discretization

Since automata operate on discrete inputs, we map the continuous RMS values to a finite alphabet:

| Symbol | Energy level | Interpretation |
|---|---|---|
| `A` | Low | Calm sections, intros |
| `B` | Medium | Verses, softer melodies |
| `C` | High | Choruses, driving beats |
| `D` | Very high | Drops, climaxes |

This gives us a string representing the energy progression of the entire song. For example:

```text
AAAAAABBBAAAACCBDD...
```

### 3. Genre Classification

We define separate NFAs for four genres:

- Rock
- EDM
- Pop
- Classical

Each NFA has its own transition rules, designed around the energy patterns we associate with that genre.

The generated string is passed through each automaton. We then calculate a score based on how well the sequence matches its transition patterns and use that to rank the genres.

We also check whether the string is accepted by each NFA under its strict language definition.

### 4. Automata Visualization

The NFAs are generated and visualized using NetworkX.

The graph for the highest-scoring genre is saved as a PNG, so we can actually see the automaton behind the prediction.

## Running it

```bash
source .venv/bin/activate
python3 main.py song.mp3
```

sample output:

```text
[*] Loading song.mp3...
[+] Total segments: 44 (~220s)

[+] Input string w:
    AAAAAABBBAAAACCBDDBCAABABBBADDCBDBDCCDBCBBCA

Genre        | Score | Match % | Strict L(M)
-------------|-------|---------|------------
Rock         | 85.0  | 75.0%   | REJECTED
EDM          | 70.5  | 70.5%   | REJECTED
Pop          | 85.0  | 75.0%   | REJECTED
Classical    | 85.0  | 75.0%   | REJECTED

========================================
PREDICTED GENRE: ROCK
========================================

[+] Saved NFA plot to rock_nfa.png
```
