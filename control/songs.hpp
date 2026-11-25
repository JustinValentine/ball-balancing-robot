// Melodies for bouncing to the beat. Each note is one impact; its length is the flight that
// follows, so the bounce heights carry the rhythm and the impacts land on the beat. Note 0 is a
// silent bounce, used where a pause would need a flight more than twice the shortest one.
// All tunes are traditional or public domain.
#pragma once

#include <cmath>
#include <string>
#include <vector>

namespace ballctl {

struct Song {
  std::string name;
  double beat;                             // Seconds per unit of length.
  std::vector<std::pair<int, double>> notes;  // MIDI note number and length in units.
};

inline const std::vector<Song>& songs() {
  static const std::vector<Song> list = {
      {"twinkle", 0.2, {{60, 1}, {60, 1}, {67, 1}, {67, 1}, {69, 1}, {69, 1}, {67, 2},
                         {65, 1}, {65, 1}, {64, 1}, {64, 1}, {62, 1}, {62, 1}, {60, 2}}},
      {"ode", 0.2, {{64, 1}, {64, 1}, {65, 1}, {67, 1}, {67, 1}, {65, 1}, {64, 1}, {62, 1},
                     {60, 1}, {60, 1}, {62, 1}, {64, 1}, {64, 1}, {62, 1}, {62, 2}}},
      {"mary", 0.2, {{64, 1}, {62, 1}, {60, 1}, {62, 1}, {64, 1}, {64, 1}, {64, 2},
                      {62, 1}, {62, 1}, {62, 2}, {64, 1}, {67, 1}, {67, 2},
                      {64, 1}, {62, 1}, {60, 1}, {62, 1}, {64, 1}, {64, 1}, {64, 1}, {64, 1},
                      {62, 1}, {62, 1}, {64, 1}, {62, 1}, {60, 2}}},
      {"jingle", 0.2, {{64, 1}, {64, 1}, {64, 2}, {64, 1}, {64, 1}, {64, 2},
                        {64, 1}, {67, 1}, {60, 1}, {62, 1}, {64, 2}}},
      // Shave and a haircut ... two bits. Lengths in eighth notes.
      {"shave", 0.2, {{60, 2}, {55, 1}, {55, 1}, {57, 2}, {55, 2}, {0, 2}, {59, 2}, {60, 2}, {0, 2}, {0, 2}}},
  };
  return list;
}

inline double note_hz(int midi) { return 440.0*std::pow(2.0, (midi-69)/12.0); }

}  // namespace ballctl
