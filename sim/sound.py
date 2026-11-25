"""Sound of the ball hitting the plate: a short click per impact, louder for a faster ball, played live or written to a WAV."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

RATE = 44100


def click(speed):
    """A ping-pong ball on acrylic: a few milliseconds of ringing shell behind a short noise burst.
    speed is the ball's impact speed in m/s and sets the loudness."""
    t = np.arange(int(.06*RATE))/RATE
    ring = sum(a*np.sin(2*np.pi*f*t)*np.exp(-t/d) for f, a, d in ((1900, 1, .006), (3400, .7, .004),
                                                                    (5600, .4, .0025), (780, .35, .012)))
    burst = np.random.default_rng(int(speed*1000)).normal(size=len(t))*np.exp(-t/.0012)*.8
    return (ring+burst)*min(1, .25+.45*speed)


def write_wav(path, samples):
    data = (np.clip(samples, -1, 1)*32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(data.tobytes())


def render(events, length):
    """Mix a click at each (time s, impact speed m/s) event into one track."""
    track = np.zeros(int(RATE*(length+1)))
    for t, speed in events:
        if t >= 0 and speed > .15:
            c = click(speed)
            i = int(t*RATE)
            n = min(len(c), len(track)-i)
            track[i:i+n] += c[:n]
    return track/max(1, np.abs(track).max()/.85)


class Player:
    """Plays clicks as they happen through afplay (macOS) or aplay (Linux)."""

    def __init__(self):
        self.command = shutil.which("afplay") if sys.platform == "darwin" else shutil.which("aplay")
        self.dir = Path(tempfile.mkdtemp(prefix="ball-clicks-"))
        self.files = {}

    def play(self, speed):
        if not self.command or speed <= .15:
            return
        level = min(9, int(speed*3))    # A few loudness steps are enough.
        if level not in self.files:
            self.files[level] = self.dir/f"{level}.wav"
            write_wav(self.files[level], click((level+.5)/3)*.9)
        args = [self.command, str(self.files[level])]
        if not args[0].endswith("afplay"):
            args.insert(1, "-q")
        subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
