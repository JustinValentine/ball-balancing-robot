"""Notes for the song impacts: a short mallet-like tone per MIDI note, played live or written to a WAV."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

RATE = 44100


def tone(midi, seconds=.45):
    f = 440*2**((midi-69)/12)
    t = np.arange(int(RATE*seconds))/RATE
    # Fundamental with a few decaying partials, like a marimba bar.
    wave_ = sum(a*np.sin(2*np.pi*f*k*t)*np.exp(-t*d) for k, a, d in ((1, 1, 7), (4, .25, 20), (10, .06, 40)))
    attack = np.minimum(1, t/.003)
    return .4*wave_*attack


def write_wav(path, samples):
    data = (np.clip(samples, -1, 1)*32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(data.tobytes())


def render(events, length):
    """Mix notes at the given (time s, MIDI) events into one track."""
    track = np.zeros(int(RATE*(length+1)))
    for t, midi in events:
        if midi > 0 and t >= 0:
            note = tone(midi)
            i = int(t*RATE)
            n = min(len(note), len(track)-i)
            track[i:i+n] += note[:n]
    return track/max(1, np.abs(track).max()/.9)


class Player:
    """Plays notes as they happen through afplay (macOS) or aplay (Linux)."""

    def __init__(self):
        self.command = shutil.which("afplay") if sys.platform == "darwin" else shutil.which("aplay")
        self.dir = Path(tempfile.mkdtemp(prefix="ball-notes-"))
        self.files = {}

    def play(self, midi):
        if not self.command or midi <= 0:
            return
        if midi not in self.files:
            self.files[midi] = self.dir/f"{midi}.wav"
            write_wav(self.files[midi], tone(midi))
        args = [self.command, str(self.files[midi])]
        if not args[0].endswith("afplay"):
            args.insert(1, "-q")
        subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
