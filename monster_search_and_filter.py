#! /usr/bin/python3

r'''###############################################################################
###################################################################################
#
#
#	Monster Search and Filter Python Module
#	Version 1.0
#
#   NOTE: Module code starts after the partial MIDI.py module @ line 1059
#
#	Based upon MIDI.py module v.6.7. by Peter Billam / pjb.com.au
#
#	Project Los Angeles
#
#	Tegridy Code 2025
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
#
#
###################################################################################
###################################################################################
#
#   Copyright 2025 Project Los Angeles / Tegridy Code
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
###################################################################################
###################################################################################
#
#	PARTIAL MIDI.py Module v.6.7. by Peter Billam
#   Please see TMIDI 2.3/tegridy-tools repo for full MIDI.py module code
# 
#   Or you can always download the latest full version from:
#
#   https://pjb.com.au/
#   https://peterbillam.gitlab.io/miditools/
#	
#	Copyright 2020 Peter Billam
#
###################################################################################
###################################################################################
#
#   Critical dependencies
#
#   !pip install cupy-cuda12x
#   !pip install numpy==1.24.4
#
###################################################################################
###################################################################################
#
#   Basic use example
#
#   import monster_search_and_filter
#
#   sigs_data_path = './Monster-MIDI-Dataset/SIGNATURES_DATA/MONSTER_SIGNATURES_DATA.pickle'
#
#   sigs_data = monster_search_and_filter.load_pickle(sigs_data_path)
#   sigs_dicts = monster_search_and_filter.load_signatures(sigs_data)
#   X, global_union = monster_search_and_filter.precompute_signatures(sigs_dicts)
#
#   monster_search_and_filter.search_and_filter(sigs_dicts, X, global_union)
#
###################################################################################
'''

###################################################################################

print('=' * 70)
print('Loading module...')
print('Please wait...')
print('=' * 70)

###################################################################################

import sys, struct, copy

Version = '6.7'
VersionDate = '20201120'

_previous_warning = ''  # 5.4
_previous_times = 0     # 5.4
_no_warning = False

#------------------------------- Encoding stuff --------------------------

def score2opus(score=None, text_encoding='ISO-8859-1'):
    r'''
The argument is a list: the first item in the list is the "ticks"
parameter, the others are the tracks. Each track is a list
of score-events, and each event is itself a list.  A score-event
is similar to an opus-event (see above), except that in a score:
 1) the times are expressed as an absolute number of ticks
    from the track's start time
 2) the pairs of 'note_on' and 'note_off' events in an "opus"
    are abstracted into a single 'note' event in a "score":
    ['note', start_time, duration, channel, pitch, velocity]
score2opus() returns a list specifying the equivalent "opus".

my_score = [
    96,
    [   # track 0:
        ['patch_change', 0, 1, 8],
        ['note', 5, 96, 1, 25, 96],
        ['note', 101, 96, 1, 29, 96]
    ],   # end of track 0
]
my_opus = score2opus(my_score)
'''
    if len(score) < 2:
        score=[1000, [],]
    tracks = copy.deepcopy(score)
    ticks = int(tracks.pop(0))
    opus_tracks = []
    for scoretrack in tracks:
        time2events = dict([])
        for scoreevent in scoretrack:
            if scoreevent[0] == 'note':
                note_on_event = ['note_on',scoreevent[1],
                 scoreevent[3],scoreevent[4],scoreevent[5]]
                note_off_event = ['note_off',scoreevent[1]+scoreevent[2],
                 scoreevent[3],scoreevent[4],scoreevent[5]]
                if time2events.get(note_on_event[1]):
                   time2events[note_on_event[1]].append(note_on_event)
                else:
                   time2events[note_on_event[1]] = [note_on_event,]
                if time2events.get(note_off_event[1]):
                   time2events[note_off_event[1]].append(note_off_event)
                else:
                   time2events[note_off_event[1]] = [note_off_event,]
                continue
            if time2events.get(scoreevent[1]):
               time2events[scoreevent[1]].append(scoreevent)
            else:
               time2events[scoreevent[1]] = [scoreevent,]

        sorted_times = []  # list of keys
        for k in time2events.keys():
            sorted_times.append(k)
        sorted_times.sort()

        sorted_events = []  # once-flattened list of values sorted by key
        for time in sorted_times:
            sorted_events.extend(time2events[time])

        abs_time = 0
        for event in sorted_events:  # convert abs times => delta times
            delta_time = event[1] - abs_time
            abs_time = event[1]
            event[1] = delta_time
        opus_tracks.append(sorted_events)
    opus_tracks.insert(0,ticks)
    _clean_up_warnings()
    return opus_tracks

#--------------------------- Decoding stuff ------------------------

def midi2opus(midi=b'', do_not_check_MIDI_signature=False):
    r'''Translates MIDI into a "opus".  For a description of the
"opus" format, see opus2midi()
'''
    my_midi=bytearray(midi)
    if len(my_midi) < 4:
        _clean_up_warnings()
        return [1000,[],]
    id = bytes(my_midi[0:4])
    if id != b'MThd':
        _warn("midi2opus: midi starts with "+str(id)+" instead of 'MThd'")
        _clean_up_warnings()
        if do_not_check_MIDI_signature == False:
          return [1000,[],]
    [length, format, tracks_expected, ticks] = struct.unpack(
     '>IHHH', bytes(my_midi[4:14]))
    if length != 6:
        _warn("midi2opus: midi header length was "+str(length)+" instead of 6")
        _clean_up_warnings()
        return [1000,[],]
    my_opus = [ticks,]
    my_midi = my_midi[14:]
    track_num = 1   # 5.1
    while len(my_midi) >= 8:
        track_type   = bytes(my_midi[0:4])
        if track_type != b'MTrk':
            #_warn('midi2opus: Warning: track #'+str(track_num)+' type is '+str(track_type)+" instead of b'MTrk'")
            pass
        [track_length] = struct.unpack('>I', my_midi[4:8])
        my_midi = my_midi[8:]
        if track_length > len(my_midi):
            _warn('midi2opus: track #'+str(track_num)+' length '+str(track_length)+' is too large')
            _clean_up_warnings()
            return my_opus   # 5.0
        my_midi_track = my_midi[0:track_length]
        my_track = _decode(my_midi_track)
        my_opus.append(my_track)
        my_midi = my_midi[track_length:]
        track_num += 1   # 5.1
    _clean_up_warnings()
    return my_opus

def opus2score(opus=[]):
    r'''For a description of the "opus" and "score" formats,
see opus2midi() and score2opus().
'''
    if len(opus) < 2:
        _clean_up_warnings()
        return [1000,[],]
    tracks = copy.deepcopy(opus)  # couple of slices probably quicker...
    ticks = int(tracks.pop(0))
    score = [ticks,]
    for opus_track in tracks:
        ticks_so_far = 0
        score_track = []
        chapitch2note_on_events = dict([])   # 4.0
        for opus_event in opus_track:
            ticks_so_far += opus_event[1]
            if opus_event[0] == 'note_off' or (opus_event[0] == 'note_on' and opus_event[4] == 0):  # 4.8
                cha = opus_event[2]
                pitch = opus_event[3]
                key = cha*128 + pitch
                if chapitch2note_on_events.get(key):
                    new_event = chapitch2note_on_events[key].pop(0)
                    new_event[2] = ticks_so_far - new_event[1]
                    score_track.append(new_event)
                elif pitch > 127:
                    pass #_warn('opus2score: note_off with no note_on, bad pitch='+str(pitch))
                else:
                    pass #_warn('opus2score: note_off with no note_on cha='+str(cha)+' pitch='+str(pitch))
            elif opus_event[0] == 'note_on':
                cha = opus_event[2]
                pitch = opus_event[3]
                key = cha*128 + pitch
                new_event = ['note',ticks_so_far,0,cha,pitch, opus_event[4]]
                if chapitch2note_on_events.get(key):
                    chapitch2note_on_events[key].append(new_event)
                else:
                    chapitch2note_on_events[key] = [new_event,]
            else:
                opus_event[1] = ticks_so_far
                score_track.append(opus_event)
        # check for unterminated notes (Oisín) -- 5.2
        for chapitch in chapitch2note_on_events:
            note_on_events = chapitch2note_on_events[chapitch]
            for new_e in note_on_events:
                new_e[2] = ticks_so_far - new_e[1]
                score_track.append(new_e)
                pass #_warn("opus2score: note_on with no note_off cha="+str(new_e[3])+' pitch='+str(new_e[4])+'; adding note_off at end')
        score.append(score_track)
    _clean_up_warnings()
    return score

def midi2score(midi=b'', do_not_check_MIDI_signature=False):
    r'''
Translates MIDI into a "score", using midi2opus() then opus2score()
'''
    return opus2score(midi2opus(midi, do_not_check_MIDI_signature))

def midi2ms_score(midi=b'', do_not_check_MIDI_signature=False):
    r'''
Translates MIDI into a "score" with one beat per second and one
tick per millisecond, using midi2opus() then to_millisecs()
then opus2score()
'''
    return opus2score(to_millisecs(midi2opus(midi, do_not_check_MIDI_signature)))

def midi2single_track_ms_score(midi_path_or_bytes, 
                                recalculate_channels = False, 
                                pass_old_timings_events= False, 
                                verbose = False, 
                                do_not_check_MIDI_signature=False
                                ):
    r'''
Translates MIDI into a single track "score" with 16 instruments and one beat per second and one
tick per millisecond
'''

    if type(midi_path_or_bytes) == bytes:
      midi_data = midi_path_or_bytes

    elif type(midi_path_or_bytes) == str:
      midi_data = open(midi_path_or_bytes, 'rb').read() 

    score = midi2score(midi_data, do_not_check_MIDI_signature)

    if recalculate_channels:

      events_matrixes = []

      itrack = 1
      events_matrixes_channels = []
      while itrack < len(score):
          events_matrix = []
          for event in score[itrack]:
              if event[0] == 'note' and event[3] != 9:
                event[3] = (16 * (itrack-1)) + event[3]
                if event[3] not in events_matrixes_channels:
                  events_matrixes_channels.append(event[3])

              events_matrix.append(event)
          events_matrixes.append(events_matrix)
          itrack += 1

      events_matrix1 = []
      for e in events_matrixes:
        events_matrix1.extend(e)

      if verbose:
        if len(events_matrixes_channels) > 16:
          print('MIDI has', len(events_matrixes_channels), 'instruments!', len(events_matrixes_channels) - 16, 'instrument(s) will be removed!')

      for e in events_matrix1:
        if e[0] == 'note' and e[3] != 9:
          if e[3] in events_matrixes_channels[:15]:
            if events_matrixes_channels[:15].index(e[3]) < 9:
              e[3] = events_matrixes_channels[:15].index(e[3])
            else:
              e[3] = events_matrixes_channels[:15].index(e[3])+1
          else:
            events_matrix1.remove(e)
        
        if e[0] in ['patch_change', 'control_change', 'channel_after_touch', 'key_after_touch', 'pitch_wheel_change'] and e[2] != 9:
          if e[2] in [e % 16 for e in events_matrixes_channels[:15]]:
            if [e % 16 for e in events_matrixes_channels[:15]].index(e[2]) < 9:
              e[2] = [e % 16 for e in events_matrixes_channels[:15]].index(e[2])
            else:
              e[2] = [e % 16 for e in events_matrixes_channels[:15]].index(e[2])+1
          else:
            events_matrix1.remove(e)
    
    else:
      events_matrix1 = []
      itrack = 1
     
      while itrack < len(score):
          for event in score[itrack]:
            events_matrix1.append(event)
          itrack += 1    

    opus = score2opus([score[0], events_matrix1])
    ms_score = opus2score(to_millisecs(opus, pass_old_timings_events=pass_old_timings_events))

    return ms_score

#------------------------ Other Transformations ---------------------

def to_millisecs(old_opus=None, desired_time_in_ms=1, pass_old_timings_events = False):
    r'''Recallibrates all the times in an "opus" to use one beat
per second and one tick per millisecond.  This makes it
hard to retrieve any information about beats or barlines,
but it does make it easy to mix different scores together.
'''
    if old_opus == None:
        return [1000 * desired_time_in_ms,[],]
    try:
        old_tpq  = int(old_opus[0])
    except IndexError:   # 5.0
        _warn('to_millisecs: the opus '+str(type(old_opus))+' has no elements')
        return [1000 * desired_time_in_ms,[],]
    new_opus = [1000 * desired_time_in_ms,]
    # 6.7 first go through building a table of set_tempos by absolute-tick
    ticks2tempo = {}
    itrack = 1
    while itrack < len(old_opus):
        ticks_so_far = 0
        for old_event in old_opus[itrack]:
            if old_event[0] == 'note':
                raise TypeError('to_millisecs needs an opus, not a score')
            ticks_so_far += old_event[1]
            if old_event[0] == 'set_tempo':
                ticks2tempo[ticks_so_far] = old_event[2]
        itrack += 1
    # then get the sorted-array of their keys
    tempo_ticks = []  # list of keys
    for k in ticks2tempo.keys():
        tempo_ticks.append(k)
    tempo_ticks.sort()
    # then go through converting to millisec, testing if the next
    # set_tempo lies before the next track-event, and using it if so.
    itrack = 1
    while itrack < len(old_opus):
        ms_per_old_tick = 400 / old_tpq  # float: will round later 6.3
        i_tempo_ticks = 0
        ticks_so_far = 0
        ms_so_far = 0.0
        previous_ms_so_far = 0.0

        if pass_old_timings_events:
          new_track = [['set_tempo',0,1000000 * desired_time_in_ms],['old_tpq', 0, old_tpq]]  # new "crochet" is 1 sec
        else:
          new_track = [['set_tempo',0,1000000 * desired_time_in_ms],]  # new "crochet" is 1 sec
        for old_event in old_opus[itrack]:
            # detect if ticks2tempo has something before this event
            # 20160702 if ticks2tempo is at the same time, leave it
            event_delta_ticks = old_event[1] * desired_time_in_ms
            if (i_tempo_ticks < len(tempo_ticks) and
              tempo_ticks[i_tempo_ticks] < (ticks_so_far + old_event[1]) * desired_time_in_ms):
                delta_ticks = tempo_ticks[i_tempo_ticks] - ticks_so_far
                ms_so_far += (ms_per_old_tick * delta_ticks * desired_time_in_ms)
                ticks_so_far = tempo_ticks[i_tempo_ticks]
                ms_per_old_tick = ticks2tempo[ticks_so_far] / (1000.0*old_tpq * desired_time_in_ms)
                i_tempo_ticks += 1
                event_delta_ticks -= delta_ticks
            new_event = copy.deepcopy(old_event)  # now handle the new event
            ms_so_far += (ms_per_old_tick * old_event[1] * desired_time_in_ms)
            new_event[1] = round(ms_so_far - previous_ms_so_far)

            if pass_old_timings_events:
              if old_event[0] != 'set_tempo':
                  previous_ms_so_far = ms_so_far
                  new_track.append(new_event)
              else:
                  new_event[0] = 'old_set_tempo'
                  previous_ms_so_far = ms_so_far
                  new_track.append(new_event)
            else:
              if old_event[0] != 'set_tempo':
                  previous_ms_so_far = ms_so_far
                  new_track.append(new_event)
            ticks_so_far += event_delta_ticks
        new_opus.append(new_track)
        itrack += 1
    _clean_up_warnings()
    return new_opus

#----------------------------- Event stuff --------------------------

_sysex2midimode = {
    "\x7E\x7F\x09\x01\xF7": 1,
    "\x7E\x7F\x09\x02\xF7": 0,
    "\x7E\x7F\x09\x03\xF7": 2,
}

# Some public-access tuples:
MIDI_events = tuple('''note_off note_on key_after_touch
control_change patch_change channel_after_touch
pitch_wheel_change'''.split())

Text_events = tuple('''text_event copyright_text_event
track_name instrument_name lyric marker cue_point text_event_08
text_event_09 text_event_0a text_event_0b text_event_0c
text_event_0d text_event_0e text_event_0f'''.split())

Nontext_meta_events = tuple('''end_track set_tempo
smpte_offset time_signature key_signature sequencer_specific
raw_meta_event sysex_f0 sysex_f7 song_position song_select
tune_request'''.split())
# unsupported: raw_data

# Actually, 'tune_request' is is F-series event, not strictly a meta-event...
Meta_events = Text_events + Nontext_meta_events
All_events  = MIDI_events + Meta_events

# And three dictionaries:
Number2patch = {   # General MIDI patch numbers:
0:'Acoustic Grand',
1:'Bright Acoustic',
2:'Electric Grand',
3:'Honky-Tonk',
4:'Electric Piano 1',
5:'Electric Piano 2',
6:'Harpsichord',
7:'Clav',
8:'Celesta',
9:'Glockenspiel',
10:'Music Box',
11:'Vibraphone',
12:'Marimba',
13:'Xylophone',
14:'Tubular Bells',
15:'Dulcimer',
16:'Drawbar Organ',
17:'Percussive Organ',
18:'Rock Organ',
19:'Church Organ',
20:'Reed Organ',
21:'Accordion',
22:'Harmonica',
23:'Tango Accordion',
24:'Acoustic Guitar(nylon)',
25:'Acoustic Guitar(steel)',
26:'Electric Guitar(jazz)',
27:'Electric Guitar(clean)',
28:'Electric Guitar(muted)',
29:'Overdriven Guitar',
30:'Distortion Guitar',
31:'Guitar Harmonics',
32:'Acoustic Bass',
33:'Electric Bass(finger)',
34:'Electric Bass(pick)',
35:'Fretless Bass',
36:'Slap Bass 1',
37:'Slap Bass 2',
38:'Synth Bass 1',
39:'Synth Bass 2',
40:'Violin',
41:'Viola',
42:'Cello',
43:'Contrabass',
44:'Tremolo Strings',
45:'Pizzicato Strings',
46:'Orchestral Harp',
47:'Timpani',
48:'String Ensemble 1',
49:'String Ensemble 2',
50:'SynthStrings 1',
51:'SynthStrings 2',
52:'Choir Aahs',
53:'Voice Oohs',
54:'Synth Voice',
55:'Orchestra Hit',
56:'Trumpet',
57:'Trombone',
58:'Tuba',
59:'Muted Trumpet',
60:'French Horn',
61:'Brass Section',
62:'SynthBrass 1',
63:'SynthBrass 2',
64:'Soprano Sax',
65:'Alto Sax',
66:'Tenor Sax',
67:'Baritone Sax',
68:'Oboe',
69:'English Horn',
70:'Bassoon',
71:'Clarinet',
72:'Piccolo',
73:'Flute',
74:'Recorder',
75:'Pan Flute',
76:'Blown Bottle',
77:'Skakuhachi',
78:'Whistle',
79:'Ocarina',
80:'Lead 1 (square)',
81:'Lead 2 (sawtooth)',
82:'Lead 3 (calliope)',
83:'Lead 4 (chiff)',
84:'Lead 5 (charang)',
85:'Lead 6 (voice)',
86:'Lead 7 (fifths)',
87:'Lead 8 (bass+lead)',
88:'Pad 1 (new age)',
89:'Pad 2 (warm)',
90:'Pad 3 (polysynth)',
91:'Pad 4 (choir)',
92:'Pad 5 (bowed)',
93:'Pad 6 (metallic)',
94:'Pad 7 (halo)',
95:'Pad 8 (sweep)',
96:'FX 1 (rain)',
97:'FX 2 (soundtrack)',
98:'FX 3 (crystal)',
99:'FX 4 (atmosphere)',
100:'FX 5 (brightness)',
101:'FX 6 (goblins)',
102:'FX 7 (echoes)',
103:'FX 8 (sci-fi)',
104:'Sitar',
105:'Banjo',
106:'Shamisen',
107:'Koto',
108:'Kalimba',
109:'Bagpipe',
110:'Fiddle',
111:'Shanai',
112:'Tinkle Bell',
113:'Agogo',
114:'Steel Drums',
115:'Woodblock',
116:'Taiko Drum',
117:'Melodic Tom',
118:'Synth Drum',
119:'Reverse Cymbal',
120:'Guitar Fret Noise',
121:'Breath Noise',
122:'Seashore',
123:'Bird Tweet',
124:'Telephone Ring',
125:'Helicopter',
126:'Applause',
127:'Gunshot',
}
Notenum2percussion = {   # General MIDI Percussion (on Channel 9):
35:'Acoustic Bass Drum',
36:'Bass Drum 1',
37:'Side Stick',
38:'Acoustic Snare',
39:'Hand Clap',
40:'Electric Snare',
41:'Low Floor Tom',
42:'Closed Hi-Hat',
43:'High Floor Tom',
44:'Pedal Hi-Hat',
45:'Low Tom',
46:'Open Hi-Hat',
47:'Low-Mid Tom',
48:'Hi-Mid Tom',
49:'Crash Cymbal 1',
50:'High Tom',
51:'Ride Cymbal 1',
52:'Chinese Cymbal',
53:'Ride Bell',
54:'Tambourine',
55:'Splash Cymbal',
56:'Cowbell',
57:'Crash Cymbal 2',
58:'Vibraslap',
59:'Ride Cymbal 2',
60:'Hi Bongo',
61:'Low Bongo',
62:'Mute Hi Conga',
63:'Open Hi Conga',
64:'Low Conga',
65:'High Timbale',
66:'Low Timbale',
67:'High Agogo',
68:'Low Agogo',
69:'Cabasa',
70:'Maracas',
71:'Short Whistle',
72:'Long Whistle',
73:'Short Guiro',
74:'Long Guiro',
75:'Claves',
76:'Hi Wood Block',
77:'Low Wood Block',
78:'Mute Cuica',
79:'Open Cuica',
80:'Mute Triangle',
81:'Open Triangle',
}

Event2channelindex = { 'note':3, 'note_off':2, 'note_on':2,
 'key_after_touch':2, 'control_change':2, 'patch_change':2,
 'channel_after_touch':2, 'pitch_wheel_change':2
}

################################################################
# The code below this line is full of frightening things, all to
# do with the actual encoding and decoding of binary MIDI data.

def _twobytes2int(byte_a):
    r'''decode a 16 bit quantity from two bytes,'''
    return (byte_a[1] | (byte_a[0] << 8))

def _int2twobytes(int_16bit):
    r'''encode a 16 bit quantity into two bytes,'''
    return bytes([(int_16bit>>8) & 0xFF, int_16bit & 0xFF])

def _read_14_bit(byte_a):
    r'''decode a 14 bit quantity from two bytes,'''
    return (byte_a[0] | (byte_a[1] << 7))

def _write_14_bit(int_14bit):
    r'''encode a 14 bit quantity into two bytes,'''
    return bytes([int_14bit & 0x7F, (int_14bit>>7) & 0x7F])

def _ber_compressed_int(integer):
    r'''BER compressed integer (not an ASN.1 BER, see perlpacktut for
details).  Its bytes represent an unsigned integer in base 128,
most significant digit first, with as few digits as possible.
Bit eight (the high bit) is set on each byte except the last.
'''
    ber = bytearray(b'')
    seven_bits = 0x7F & integer
    ber.insert(0, seven_bits)  # XXX surely should convert to a char ?
    integer >>= 7
    while integer > 0:
        seven_bits = 0x7F & integer
        ber.insert(0, 0x80|seven_bits)  # XXX surely should convert to a char ?
        integer >>= 7
    return ber

def _unshift_ber_int(ba):
    r'''Given a bytearray, returns a tuple of (the ber-integer at the
start, and the remainder of the bytearray).
'''
    if not len(ba):  # 6.7
        _warn('_unshift_ber_int: no integer found')
        return ((0, b""))
    byte = ba[0]
    ba = ba[1:]
    integer = 0
    while True:
        integer += (byte & 0x7F)
        if not (byte & 0x80):
            return ((integer, ba))
        if not len(ba):
            _warn('_unshift_ber_int: no end-of-integer found')
            return ((0, ba))
        byte = ba[0]
        ba = ba[1:]
        integer <<= 7


def _clean_up_warnings():  # 5.4
    # Call this before returning from any publicly callable function
    # whenever there's a possibility that a warning might have been printed
    # by the function, or by any private functions it might have called.
    if _no_warning:
        return
    global _previous_times
    global _previous_warning
    if _previous_times > 1:
        # E:1176, 0: invalid syntax (<string>, line 1176) (syntax-error) ???
        # print('  previous message repeated '+str(_previous_times)+' times', file=sys.stderr)
        # 6.7
        sys.stderr.write('  previous message repeated {0} times\n'.format(_previous_times))
    elif _previous_times > 0:
        sys.stderr.write('  previous message repeated\n')
    _previous_times = 0
    _previous_warning = ''


def _warn(s=''):
    if _no_warning:
        return
    global _previous_times
    global _previous_warning
    if s == _previous_warning:  # 5.4
        _previous_times = _previous_times + 1
    else:
        _clean_up_warnings()
        sys.stderr.write(str(s) + "\n")
        _previous_warning = s


def _some_text_event(which_kind=0x01, text=b'some_text', text_encoding='ISO-8859-1'):
    if str(type(text)).find("'str'") >= 0:  # 6.4 test for back-compatibility
        data = bytes(text, encoding=text_encoding)
    else:
        data = bytes(text)
    return b'\xFF' + bytes((which_kind,)) + _ber_compressed_int(len(data)) + data


def _consistentise_ticks(scores):  # 3.6
    # used by mix_scores, merge_scores, concatenate_scores
    if len(scores) == 1:
        return copy.deepcopy(scores)
    are_consistent = True
    ticks = scores[0][0]
    iscore = 1
    while iscore < len(scores):
        if scores[iscore][0] != ticks:
            are_consistent = False
            break
        iscore += 1
    if are_consistent:
        return copy.deepcopy(scores)
    new_scores = []
    iscore = 0
    while iscore < len(scores):
        score = scores[iscore]
        new_scores.append(opus2score(to_millisecs(score2opus(score))))
        iscore += 1
    return new_scores


###########################################################################
def _decode(trackdata=b'', exclude=None, include=None,
            event_callback=None, exclusive_event_callback=None, no_eot_magic=False):
    r'''Decodes MIDI track data into an opus-style list of events.
The options:
  'exclude' is a list of event types which will be ignored SHOULD BE A SET
  'include' (and no exclude), makes exclude a list
       of all possible events, /minus/ what include specifies
  'event_callback' is a coderef
  'exclusive_event_callback' is a coderef
'''
    trackdata = bytearray(trackdata)
    if exclude == None:
        exclude = []
    if include == None:
        include = []
    if include and not exclude:
        exclude = All_events
    include = set(include)
    exclude = set(exclude)

    # Pointer = 0;  not used here; we eat through the bytearray instead.
    event_code = -1;  # used for running status
    event_count = 0;
    events = []

    while (len(trackdata)):
        # loop while there's anything to analyze ...
        eot = False  # When True, the event registrar aborts this loop
        event_count += 1

        E = []
        # E for events - we'll feed it to the event registrar at the end.

        # Slice off the delta time code, and analyze it
        [time, trackdata] = _unshift_ber_int(trackdata)

        # Now let's see what we can make of the command
        first_byte = trackdata[0] & 0xFF
        trackdata = trackdata[1:]
        if (first_byte < 0xF0):  # It's a MIDI event
            if (first_byte & 0x80):
                event_code = first_byte
            else:
                # It wants running status; use last event_code value
                trackdata.insert(0, first_byte)
                if (event_code == -1):
                    _warn("Running status not set; Aborting track.")
                    return []

            command = event_code & 0xF0
            channel = event_code & 0x0F

            if (command == 0xF6):  # 0-byte argument
                pass
            elif (command == 0xC0 or command == 0xD0):  # 1-byte argument
                parameter = trackdata[0]  # could be B
                trackdata = trackdata[1:]
            else:  # 2-byte argument could be BB or 14-bit
                parameter = (trackdata[0], trackdata[1])
                trackdata = trackdata[2:]

            #################################################################
            # MIDI events

            if (command == 0x80):
                if 'note_off' in exclude:
                    continue
                E = ['note_off', time, channel, parameter[0], parameter[1]]
            elif (command == 0x90):
                if 'note_on' in exclude:
                    continue
                E = ['note_on', time, channel, parameter[0], parameter[1]]
            elif (command == 0xA0):
                if 'key_after_touch' in exclude:
                    continue
                E = ['key_after_touch', time, channel, parameter[0], parameter[1]]
            elif (command == 0xB0):
                if 'control_change' in exclude:
                    continue
                E = ['control_change', time, channel, parameter[0], parameter[1]]
            elif (command == 0xC0):
                if 'patch_change' in exclude:
                    continue
                E = ['patch_change', time, channel, parameter]
            elif (command == 0xD0):
                if 'channel_after_touch' in exclude:
                    continue
                E = ['channel_after_touch', time, channel, parameter]
            elif (command == 0xE0):
                if 'pitch_wheel_change' in exclude:
                    continue
                E = ['pitch_wheel_change', time, channel,
                     _read_14_bit(parameter) - 0x2000]
            else:
                _warn("Shouldn't get here; command=" + hex(command))

        elif (first_byte == 0xFF):  # It's a Meta-Event! ##################
            # [command, length, remainder] =
            #    unpack("xCwa*", substr(trackdata, $Pointer, 6));
            # Pointer += 6 - len(remainder);
            #    # Move past JUST the length-encoded.
            command = trackdata[0] & 0xFF
            trackdata = trackdata[1:]
            [length, trackdata] = _unshift_ber_int(trackdata)
            if (command == 0x00):
                if (length == 2):
                    E = ['set_sequence_number', time, _twobytes2int(trackdata)]
                else:
                    _warn('set_sequence_number: length must be 2, not ' + str(length))
                    E = ['set_sequence_number', time, 0]

            elif command >= 0x01 and command <= 0x0f:  # Text events
                # 6.2 take it in bytes; let the user get the right encoding.
                # text_str = trackdata[0:length].decode('ascii','ignore')
                # text_str = trackdata[0:length].decode('ISO-8859-1')
                # 6.4 take it in bytes; let the user get the right encoding.
                text_data = bytes(trackdata[0:length])  # 6.4
                # Defined text events
                if (command == 0x01):
                    E = ['text_event', time, text_data]
                elif (command == 0x02):
                    E = ['copyright_text_event', time, text_data]
                elif (command == 0x03):
                    E = ['track_name', time, text_data]
                elif (command == 0x04):
                    E = ['instrument_name', time, text_data]
                elif (command == 0x05):
                    E = ['lyric', time, text_data]
                elif (command == 0x06):
                    E = ['marker', time, text_data]
                elif (command == 0x07):
                    E = ['cue_point', time, text_data]
                # Reserved but apparently unassigned text events
                elif (command == 0x08):
                    E = ['text_event_08', time, text_data]
                elif (command == 0x09):
                    E = ['text_event_09', time, text_data]
                elif (command == 0x0a):
                    E = ['text_event_0a', time, text_data]
                elif (command == 0x0b):
                    E = ['text_event_0b', time, text_data]
                elif (command == 0x0c):
                    E = ['text_event_0c', time, text_data]
                elif (command == 0x0d):
                    E = ['text_event_0d', time, text_data]
                elif (command == 0x0e):
                    E = ['text_event_0e', time, text_data]
                elif (command == 0x0f):
                    E = ['text_event_0f', time, text_data]

            # Now the sticky events -------------------------------------
            elif (command == 0x2F):
                E = ['end_track', time]
                # The code for handling this, oddly, comes LATER,
                # in the event registrar.
            elif (command == 0x51):  # DTime, Microseconds/Crochet
                if length != 3:
                    _warn('set_tempo event, but length=' + str(length))
                E = ['set_tempo', time,
                     struct.unpack(">I", b'\x00' + trackdata[0:3])[0]]
            elif (command == 0x54):
                if length != 5:  # DTime, HR, MN, SE, FR, FF
                    _warn('smpte_offset event, but length=' + str(length))
                E = ['smpte_offset', time] + list(struct.unpack(">BBBBB", trackdata[0:5]))
            elif (command == 0x58):
                if length != 4:  # DTime, NN, DD, CC, BB
                    _warn('time_signature event, but length=' + str(length))
                E = ['time_signature', time] + list(trackdata[0:4])
            elif (command == 0x59):
                if length != 2:  # DTime, SF(signed), MI
                    _warn('key_signature event, but length=' + str(length))
                E = ['key_signature', time] + list(struct.unpack(">bB", trackdata[0:2]))
            elif (command == 0x7F):  # 6.4
                E = ['sequencer_specific', time, bytes(trackdata[0:length])]
            else:
                E = ['raw_meta_event', time, command,
                     bytes(trackdata[0:length])]  # 6.0
                # "[uninterpretable meta-event command of length length]"
                # DTime, Command, Binary Data
                # It's uninterpretable; record it as raw_data.

            # Pointer += length; #  Now move Pointer
            trackdata = trackdata[length:]

        ######################################################################
        elif (first_byte == 0xF0 or first_byte == 0xF7):
            # Note that sysexes in MIDI /files/ are different than sysexes
            # in MIDI transmissions!! The vast majority of system exclusive
            # messages will just use the F0 format. For instance, the
            # transmitted message F0 43 12 00 07 F7 would be stored in a
            # MIDI file as F0 05 43 12 00 07 F7. As mentioned above, it is
            # required to include the F7 at the end so that the reader of the
            # MIDI file knows that it has read the entire message. (But the F7
            # is omitted if this is a non-final block in a multiblock sysex;
            # but the F7 (if there) is counted in the message's declared
            # length, so we don't have to think about it anyway.)
            # command = trackdata.pop(0)
            [length, trackdata] = _unshift_ber_int(trackdata)
            if first_byte == 0xF0:
                # 20091008 added ISO-8859-1 to get an 8-bit str
                # 6.4 return bytes instead
                E = ['sysex_f0', time, bytes(trackdata[0:length])]
            else:
                E = ['sysex_f7', time, bytes(trackdata[0:length])]
            trackdata = trackdata[length:]

        ######################################################################
        # Now, the MIDI file spec says:
        #  <track data> = <MTrk event>+
        #  <MTrk event> = <delta-time> <event>
        #  <event> = <MIDI event> | <sysex event> | <meta-event>
        # I know that, on the wire, <MIDI event> can include note_on,
        # note_off, and all the other 8x to Ex events, AND Fx events
        # other than F0, F7, and FF -- namely, <song position msg>,
        # <song select msg>, and <tune request>.
        #
        # Whether these can occur in MIDI files is not clear specified
        # from the MIDI file spec.  So, I'm going to assume that
        # they CAN, in practice, occur.  I don't know whether it's
        # proper for you to actually emit these into a MIDI file.

        elif (first_byte == 0xF2):  # DTime, Beats
            #  <song position msg> ::=     F2 <data pair>
            E = ['song_position', time, _read_14_bit(trackdata[:2])]
            trackdata = trackdata[2:]

        elif (first_byte == 0xF3):  # <song select msg> ::= F3 <data singlet>
            # E = ['song_select', time, struct.unpack('>B',trackdata.pop(0))[0]]
            E = ['song_select', time, trackdata[0]]
            trackdata = trackdata[1:]
            # DTime, Thing (what?! song number?  whatever ...)

        elif (first_byte == 0xF6):  # DTime
            E = ['tune_request', time]
            # What would a tune request be doing in a MIDI /file/?

            #########################################################
            # ADD MORE META-EVENTS HERE.  TODO:
            # f1 -- MTC Quarter Frame Message. One data byte follows
            #     the Status; it's the time code value, from 0 to 127.
            # f8 -- MIDI clock.    no data.
            # fa -- MIDI start.    no data.
            # fb -- MIDI continue. no data.
            # fc -- MIDI stop.     no data.
            # fe -- Active sense.  no data.
            # f4 f5 f9 fd -- unallocated

            r'''
        elif (first_byte > 0xF0) { # Some unknown kinda F-series event ####
            # Here we only produce a one-byte piece of raw data.
            # But the encoder for 'raw_data' accepts any length of it.
            E = [ 'raw_data',
                         time, substr(trackdata,Pointer,1) ]
            # DTime and the Data (in this case, the one Event-byte)
            ++Pointer;  # itself

'''
        elif first_byte > 0xF0:  # Some unknown F-series event
            # Here we only produce a one-byte piece of raw data.
            # E = ['raw_data', time, bytest(trackdata[0])]   # 6.4
            E = ['raw_data', time, trackdata[0]]  # 6.4 6.7
            trackdata = trackdata[1:]
        else:  # Fallthru.
            _warn("Aborting track.  Command-byte first_byte=" + hex(first_byte))
            break
        # End of the big if-group

        ######################################################################
        #  THE EVENT REGISTRAR...
        if E and (E[0] == 'end_track'):
            # This is the code for exceptional handling of the EOT event.
            eot = True
            if not no_eot_magic:
                if E[1] > 0:  # a null text-event to carry the delta-time
                    E = ['text_event', E[1], '']
                else:
                    E = []  # EOT with a delta-time of 0; ignore it.

        if E and not (E[0] in exclude):
            # if ( $exclusive_event_callback ):
            #    &{ $exclusive_event_callback }( @E );
            # else:
            #    &{ $event_callback }( @E ) if $event_callback;
            events.append(E)
        if eot:
            break

    # End of the big "Event" while-block

    return events

###################################################################################
###################################################################################
###################################################################################

import os

import datetime

import copy

from datetime import datetime

import secrets

import random

import pickle

import tqdm

import multiprocessing

from collections import Counter

from itertools import combinations

import sys

import statistics
import math

from collections import defaultdict

try:
    import cupy as np
    print('CuPy is found!')
    print('Will use CuPy and GPU for processing!')

except:
    import numpy as np
    print('Could not load CuPy!')
    print('Will use NumPy and CPU for processing!')
    
import shutil

print('=' * 70)

###################################################################################
###################################################################################

ALL_CHORDS_SORTED = [[0], [0, 2], [0, 3], [0, 4], [0, 2, 4], [0, 5], [0, 2, 5], [0, 3, 5], [0, 6],
                    [0, 2, 6], [0, 3, 6], [0, 4, 6], [0, 2, 4, 6], [0, 7], [0, 2, 7], [0, 3, 7],
                    [0, 4, 7], [0, 5, 7], [0, 2, 4, 7], [0, 2, 5, 7], [0, 3, 5, 7], [0, 8],
                    [0, 2, 8], [0, 3, 8], [0, 4, 8], [0, 5, 8], [0, 6, 8], [0, 2, 4, 8],
                    [0, 2, 5, 8], [0, 2, 6, 8], [0, 3, 5, 8], [0, 3, 6, 8], [0, 4, 6, 8],
                    [0, 2, 4, 6, 8], [0, 9], [0, 2, 9], [0, 3, 9], [0, 4, 9], [0, 5, 9], [0, 6, 9],
                    [0, 7, 9], [0, 2, 4, 9], [0, 2, 5, 9], [0, 2, 6, 9], [0, 2, 7, 9],
                    [0, 3, 5, 9], [0, 3, 6, 9], [0, 3, 7, 9], [0, 4, 6, 9], [0, 4, 7, 9],
                    [0, 5, 7, 9], [0, 2, 4, 6, 9], [0, 2, 4, 7, 9], [0, 2, 5, 7, 9],
                    [0, 3, 5, 7, 9], [0, 10], [0, 2, 10], [0, 3, 10], [0, 4, 10], [0, 5, 10],
                    [0, 6, 10], [0, 7, 10], [0, 8, 10], [0, 2, 4, 10], [0, 2, 5, 10],
                    [0, 2, 6, 10], [0, 2, 7, 10], [0, 2, 8, 10], [0, 3, 5, 10], [0, 3, 6, 10],
                    [0, 3, 7, 10], [0, 3, 8, 10], [0, 4, 6, 10], [0, 4, 7, 10], [0, 4, 8, 10],
                    [0, 5, 7, 10], [0, 5, 8, 10], [0, 6, 8, 10], [0, 2, 4, 6, 10],
                    [0, 2, 4, 7, 10], [0, 2, 4, 8, 10], [0, 2, 5, 7, 10], [0, 2, 5, 8, 10],
                    [0, 2, 6, 8, 10], [0, 3, 5, 7, 10], [0, 3, 5, 8, 10], [0, 3, 6, 8, 10],
                    [0, 4, 6, 8, 10], [0, 2, 4, 6, 8, 10], [1], [1, 3], [1, 4], [1, 5], [1, 3, 5],
                    [1, 6], [1, 3, 6], [1, 4, 6], [1, 7], [1, 3, 7], [1, 4, 7], [1, 5, 7],
                    [1, 3, 5, 7], [1, 8], [1, 3, 8], [1, 4, 8], [1, 5, 8], [1, 6, 8], [1, 3, 5, 8],
                    [1, 3, 6, 8], [1, 4, 6, 8], [1, 9], [1, 3, 9], [1, 4, 9], [1, 5, 9], [1, 6, 9],
                    [1, 7, 9], [1, 3, 5, 9], [1, 3, 6, 9], [1, 3, 7, 9], [1, 4, 6, 9],
                    [1, 4, 7, 9], [1, 5, 7, 9], [1, 3, 5, 7, 9], [1, 10], [1, 3, 10], [1, 4, 10],
                    [1, 5, 10], [1, 6, 10], [1, 7, 10], [1, 8, 10], [1, 3, 5, 10], [1, 3, 6, 10],
                    [1, 3, 7, 10], [1, 3, 8, 10], [1, 4, 6, 10], [1, 4, 7, 10], [1, 4, 8, 10],
                    [1, 5, 7, 10], [1, 5, 8, 10], [1, 6, 8, 10], [1, 3, 5, 7, 10],
                    [1, 3, 5, 8, 10], [1, 3, 6, 8, 10], [1, 4, 6, 8, 10], [1, 11], [1, 3, 11],
                    [1, 4, 11], [1, 5, 11], [1, 6, 11], [1, 7, 11], [1, 8, 11], [1, 9, 11],
                    [1, 3, 5, 11], [1, 3, 6, 11], [1, 3, 7, 11], [1, 3, 8, 11], [1, 3, 9, 11],
                    [1, 4, 6, 11], [1, 4, 7, 11], [1, 4, 8, 11], [1, 4, 9, 11], [1, 5, 7, 11],
                    [1, 5, 8, 11], [1, 5, 9, 11], [1, 6, 8, 11], [1, 6, 9, 11], [1, 7, 9, 11],
                    [1, 3, 5, 7, 11], [1, 3, 5, 8, 11], [1, 3, 5, 9, 11], [1, 3, 6, 8, 11],
                    [1, 3, 6, 9, 11], [1, 3, 7, 9, 11], [1, 4, 6, 8, 11], [1, 4, 6, 9, 11],
                    [1, 4, 7, 9, 11], [1, 5, 7, 9, 11], [1, 3, 5, 7, 9, 11], [2], [2, 4], [2, 5],
                    [2, 6], [2, 4, 6], [2, 7], [2, 4, 7], [2, 5, 7], [2, 8], [2, 4, 8], [2, 5, 8],
                    [2, 6, 8], [2, 4, 6, 8], [2, 9], [2, 4, 9], [2, 5, 9], [2, 6, 9], [2, 7, 9],
                    [2, 4, 6, 9], [2, 4, 7, 9], [2, 5, 7, 9], [2, 10], [2, 4, 10], [2, 5, 10],
                    [2, 6, 10], [2, 7, 10], [2, 8, 10], [2, 4, 6, 10], [2, 4, 7, 10],
                    [2, 4, 8, 10], [2, 5, 7, 10], [2, 5, 8, 10], [2, 6, 8, 10], [2, 4, 6, 8, 10],
                    [2, 11], [2, 4, 11], [2, 5, 11], [2, 6, 11], [2, 7, 11], [2, 8, 11],
                    [2, 9, 11], [2, 4, 6, 11], [2, 4, 7, 11], [2, 4, 8, 11], [2, 4, 9, 11],
                    [2, 5, 7, 11], [2, 5, 8, 11], [2, 5, 9, 11], [2, 6, 8, 11], [2, 6, 9, 11],
                    [2, 7, 9, 11], [2, 4, 6, 8, 11], [2, 4, 6, 9, 11], [2, 4, 7, 9, 11],
                    [2, 5, 7, 9, 11], [3], [3, 5], [3, 6], [3, 7], [3, 5, 7], [3, 8], [3, 5, 8],
                    [3, 6, 8], [3, 9], [3, 5, 9], [3, 6, 9], [3, 7, 9], [3, 5, 7, 9], [3, 10],
                    [3, 5, 10], [3, 6, 10], [3, 7, 10], [3, 8, 10], [3, 5, 7, 10], [3, 5, 8, 10],
                    [3, 6, 8, 10], [3, 11], [3, 5, 11], [3, 6, 11], [3, 7, 11], [3, 8, 11],
                    [3, 9, 11], [3, 5, 7, 11], [3, 5, 8, 11], [3, 5, 9, 11], [3, 6, 8, 11],
                    [3, 6, 9, 11], [3, 7, 9, 11], [3, 5, 7, 9, 11], [4], [4, 6], [4, 7], [4, 8],
                    [4, 6, 8], [4, 9], [4, 6, 9], [4, 7, 9], [4, 10], [4, 6, 10], [4, 7, 10],
                    [4, 8, 10], [4, 6, 8, 10], [4, 11], [4, 6, 11], [4, 7, 11], [4, 8, 11],
                    [4, 9, 11], [4, 6, 8, 11], [4, 6, 9, 11], [4, 7, 9, 11], [5], [5, 7], [5, 8],
                    [5, 9], [5, 7, 9], [5, 10], [5, 7, 10], [5, 8, 10], [5, 11], [5, 7, 11],
                    [5, 8, 11], [5, 9, 11], [5, 7, 9, 11], [6], [6, 8], [6, 9], [6, 10],
                    [6, 8, 10], [6, 11], [6, 8, 11], [6, 9, 11], [7], [7, 9], [7, 10], [7, 11],
                    [7, 9, 11], [8], [8, 10], [8, 11], [9], [9, 11], [10], [11]]

###################################################################################

ALL_CHORDS_FULL = [[0], [0, 3], [0, 3, 5], [0, 3, 5, 8], [0, 3, 5, 9], [0, 3, 5, 10], [0, 3, 6],
                  [0, 3, 6, 9], [0, 3, 6, 10], [0, 3, 7], [0, 3, 7, 10], [0, 3, 8], [0, 3, 9],
                  [0, 3, 10], [0, 4], [0, 4, 6], [0, 4, 6, 9], [0, 4, 6, 10], [0, 4, 7],
                  [0, 4, 7, 10], [0, 4, 8], [0, 4, 9], [0, 4, 10], [0, 5], [0, 5, 8], [0, 5, 9],
                  [0, 5, 10], [0, 6], [0, 6, 9], [0, 6, 10], [0, 7], [0, 7, 10], [0, 8], [0, 9],
                  [0, 10], [1], [1, 4], [1, 4, 6], [1, 4, 6, 9], [1, 4, 6, 10], [1, 4, 6, 11],
                  [1, 4, 7], [1, 4, 7, 10], [1, 4, 7, 11], [1, 4, 8], [1, 4, 8, 11], [1, 4, 9],
                  [1, 4, 10], [1, 4, 11], [1, 5], [1, 5, 8], [1, 5, 8, 11], [1, 5, 9],
                  [1, 5, 10], [1, 5, 11], [1, 6], [1, 6, 9], [1, 6, 10], [1, 6, 11], [1, 7],
                  [1, 7, 10], [1, 7, 11], [1, 8], [1, 8, 11], [1, 9], [1, 10], [1, 11], [2],
                  [2, 5], [2, 5, 8], [2, 5, 8, 11], [2, 5, 9], [2, 5, 10], [2, 5, 11], [2, 6],
                  [2, 6, 9], [2, 6, 10], [2, 6, 11], [2, 7], [2, 7, 10], [2, 7, 11], [2, 8],
                  [2, 8, 11], [2, 9], [2, 10], [2, 11], [3], [3, 5], [3, 5, 8], [3, 5, 8, 11],
                  [3, 5, 9], [3, 5, 10], [3, 5, 11], [3, 6], [3, 6, 9], [3, 6, 10], [3, 6, 11],
                  [3, 7], [3, 7, 10], [3, 7, 11], [3, 8], [3, 8, 11], [3, 9], [3, 10], [3, 11],
                  [4], [4, 6], [4, 6, 9], [4, 6, 10], [4, 6, 11], [4, 7], [4, 7, 10], [4, 7, 11],
                  [4, 8], [4, 8, 11], [4, 9], [4, 10], [4, 11], [5], [5, 8], [5, 8, 11], [5, 9],
                  [5, 10], [5, 11], [6], [6, 9], [6, 10], [6, 11], [7], [7, 10], [7, 11], [8],
                  [8, 11], [9], [10], [11]]

###################################################################################
###################################################################################

def create_files_list(datasets_paths=['./'],
                      files_exts=['.mid', '.midi', '.kar', '.MID', '.MIDI', '.KAR'],
                      randomize_files_list=True,
                      verbose=True
                     ):
    if verbose:
        print('=' * 70)
        print('Searching for files...')
        print('This may take a while on a large dataset in particular...')
        print('=' * 70)

    filez_set = defaultdict(None)

    files_exts = tuple(files_exts)
    
    for dataset_addr in tqdm.tqdm(datasets_paths):
        for dirpath, dirnames, filenames in os.walk(dataset_addr):
            for file in filenames:
                if file not in filez_set and file.endswith(files_exts):
                    filez_set[os.path.join(dirpath, file)] = None
    
    filez = list(filez_set.keys())

    if verbose:
        print('Done!')
        print('=' * 70)
    
    if filez:
        if randomize_files_list:
            
            if verbose:
                print('Randomizing file list...')
                
            random.shuffle(filez)
            
            if verbose:
                print('Done!')
                print('=' * 70)
                
        if verbose:
            print('Found', len(filez), 'files.')
            print('=' * 70)
 
    else:
        if verbose:
            print('Could not find any files...')
            print('Please check dataset dirs and files extensions...')
            print('=' * 70)
        
    return filez

###################################################################################

def check_and_fix_tones_chord(tones_chord, use_full_chords=True):

  tones_chord_combs = [list(comb) for i in range(len(tones_chord), 0, -1) for comb in combinations(tones_chord, i)]

  if use_full_chords:
    CHORDS = ALL_CHORDS_FULL

  else:
    CHORDS = ALL_CHORDS_SORTED

  for c in tones_chord_combs:
    if c in CHORDS:
      checked_tones_chord = c
      break

  return sorted(checked_tones_chord)

###################################################################################

def chordify_score(score,
                  return_choridfied_score=True,
                  return_detected_score_information=False
                  ):

    if score:
    
      num_tracks = 1
      single_track_score = []
      score_num_ticks = 0

      if type(score[0]) == int and len(score) > 1:

        score_type = 'MIDI_PY'
        score_num_ticks = score[0]

        while num_tracks < len(score):
            for event in score[num_tracks]:
              single_track_score.append(event)
            num_tracks += 1
      
      else:
        score_type = 'CUSTOM'
        single_track_score = score

      if single_track_score and single_track_score[0]:
        
        try:

          if type(single_track_score[0][0]) == str or single_track_score[0][0] == 'note':
            single_track_score.sort(key = lambda x: x[1])
            score_timings = [s[1] for s in single_track_score]
          else:
            score_timings = [s[0] for s in single_track_score]

          is_score_time_absolute = lambda sct: all(x <= y for x, y in zip(sct, sct[1:]))

          score_timings_type = ''

          if is_score_time_absolute(score_timings):
            score_timings_type = 'ABS'

            chords = []
            cho = []

            if score_type == 'MIDI_PY':
              pe = single_track_score[0]
            else:
              pe = single_track_score[0]

            for e in single_track_score:
              
              if score_type == 'MIDI_PY':
                time = e[1]
                ptime = pe[1]
              else:
                time = e[0]
                ptime = pe[0]

              if time == ptime:
                cho.append(e)
              
              else:
                if len(cho) > 0:
                  chords.append(cho)
                cho = []
                cho.append(e)

              pe = e

            if len(cho) > 0:
              chords.append(cho)

          else:
            score_timings_type = 'REL'
            
            chords = []
            cho = []

            for e in single_track_score:
              
              if score_type == 'MIDI_PY':
                time = e[1]
              else:
                time = e[0]

              if time == 0:
                cho.append(e)
              
              else:
                if len(cho) > 0:
                  chords.append(cho)
                cho = []
                cho.append(e)

            if len(cho) > 0:
              chords.append(cho)

          requested_data = []

          if return_detected_score_information:
            
            detected_score_information = []

            detected_score_information.append(['Score type', score_type])
            detected_score_information.append(['Score timings type', score_timings_type])
            detected_score_information.append(['Score tpq', score_num_ticks])
            detected_score_information.append(['Score number of tracks', num_tracks])
            
            requested_data.append(detected_score_information)

          if return_choridfied_score and return_detected_score_information:
            requested_data.append(chords)

          if return_choridfied_score and not return_detected_score_information:
            requested_data.extend(chords)

          return requested_data

        except Exception as e:
          print('Error!')
          print('Check score for consistency and compatibility!')
          print('Exception detected:', e)

      else:
        return None

    else:
      return None

###################################################################################

def augment_enhanced_score_notes(enhanced_score_notes,
                                  timings_divider=16,
                                  full_sorting=True,
                                  timings_shift=0,
                                  pitch_shift=0,
                                  ceil_timings=False,
                                  round_timings=False,
                                  legacy_timings=True,
                                  sort_drums_last=False
                                ):

    esn = copy.deepcopy(enhanced_score_notes)

    pe = enhanced_score_notes[0]

    abs_time = max(0, int(enhanced_score_notes[0][1] / timings_divider))

    for i, e in enumerate(esn):
      
      dtime = (e[1] / timings_divider) - (pe[1] / timings_divider)

      if round_timings:
        dtime = round(dtime)
      
      else:
        if ceil_timings:
          dtime = math.ceil(dtime)
        
        else:
          dtime = int(dtime)

      if legacy_timings:
        abs_time = int(e[1] / timings_divider) + timings_shift

      else:
        abs_time += dtime

      e[1] = max(0, abs_time + timings_shift)

      if round_timings:
        e[2] = max(1, round(e[2] / timings_divider)) + timings_shift
      
      else:
        if ceil_timings:
          e[2] = max(1, math.ceil(e[2] / timings_divider)) + timings_shift
        else:
          e[2] = max(1, int(e[2] / timings_divider)) + timings_shift
      
      e[4] = max(1, min(127, e[4] + pitch_shift))

      pe = enhanced_score_notes[i]

    if full_sorting:

      # Sorting by patch, reverse pitch and start-time
      esn.sort(key=lambda x: x[6])
      esn.sort(key=lambda x: x[4], reverse=True)
      esn.sort(key=lambda x: x[1])
      
    if sort_drums_last:
        esn.sort(key=lambda x: (x[1], -x[4], x[6]) if x[6] != 128 else (x[1], x[6], -x[4]))

    return esn

###################################################################################

def advanced_score_processor(raw_score, 
                              patches_to_analyze=list(range(129)), 
                              return_score_analysis=False,
                              return_enhanced_score=False,
                              return_enhanced_score_notes=False,
                              return_enhanced_monophonic_melody=False,
                              return_chordified_enhanced_score=False,
                              return_chordified_enhanced_score_with_lyrics=False,
                              return_score_tones_chords=False,
                              return_text_and_lyric_events=False
                            ):

  '''TMIDIX Advanced Score Processor'''

  # Score data types detection

  if raw_score and type(raw_score) == list:

      num_ticks = 0
      num_tracks = 1

      basic_single_track_score = []

      if type(raw_score[0]) != int:
        if len(raw_score[0]) < 5 and type(raw_score[0][0]) != str:
          return ['Check score for errors and compatibility!']

        else:
          basic_single_track_score = copy.deepcopy(raw_score)
      
      else:
        num_ticks = raw_score[0]
        while num_tracks < len(raw_score):
            for event in raw_score[num_tracks]:
              ev = copy.deepcopy(event)
              basic_single_track_score.append(ev)
            num_tracks += 1

      basic_single_track_score.sort(key=lambda x: x[4] if x[0] == 'note' else 128, reverse=True)
      basic_single_track_score.sort(key=lambda x: x[1])

      enhanced_single_track_score = []
      patches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      all_score_patches = []
      num_patch_changes = 0

      for event in basic_single_track_score:
        if event[0] == 'patch_change':
              patches[event[2]] = event[3]
              enhanced_single_track_score.append(event)
              num_patch_changes += 1

        if event[0] == 'note':
            if event[3] != 9:
              event.extend([patches[event[3]]])
              all_score_patches.extend([patches[event[3]]])
            else:
              event.extend([128])
              all_score_patches.extend([128])

            if enhanced_single_track_score:
                if (event[1] == enhanced_single_track_score[-1][1]):
                    if ([event[3], event[4]] != enhanced_single_track_score[-1][3:5]):
                        enhanced_single_track_score.append(event)
                else:
                    enhanced_single_track_score.append(event)

            else:
                enhanced_single_track_score.append(event)

        if event[0] not in ['note', 'patch_change']:
          enhanced_single_track_score.append(event)

      enhanced_single_track_score.sort(key=lambda x: x[6] if x[0] == 'note' else -1)
      enhanced_single_track_score.sort(key=lambda x: x[4] if x[0] == 'note' else 128, reverse=True)
      enhanced_single_track_score.sort(key=lambda x: x[1])

      # Analysis and chordification

      cscore = []
      cescore = []
      chords_tones = []
      tones_chords = []
      all_tones = []
      all_chords_good = True
      bad_chords = []
      bad_chords_count = 0
      score_notes = []
      score_pitches = []
      score_patches = []
      num_text_events = 0
      num_lyric_events = 0
      num_other_events = 0
      text_and_lyric_events = []
      text_and_lyric_events_latin = None

      analysis = {}

      score_notes = [s for s in enhanced_single_track_score if s[0] == 'note' and s[6] in patches_to_analyze]
      score_patches = [sn[6] for sn in score_notes]

      if return_text_and_lyric_events:
        text_and_lyric_events = [e for e in enhanced_single_track_score if e[0] in ['text_event', 'lyric']]
        
        if text_and_lyric_events:
          text_and_lyric_events_latin = True
          for e in text_and_lyric_events:
            try:
              tle = str(e[2].decode())
            except:
              tle = str(e[2])

            for c in tle:
              if not 0 <= ord(c) < 128:
                text_and_lyric_events_latin = False

      if (return_chordified_enhanced_score or return_score_analysis) and any(elem in patches_to_analyze for elem in score_patches):

        cescore = chordify_score([num_ticks, enhanced_single_track_score])

        if return_score_analysis:

          cscore = chordify_score(score_notes)
          
          score_pitches = [sn[4] for sn in score_notes]
          
          text_events = [e for e in enhanced_single_track_score if e[0] == 'text_event']
          num_text_events = len(text_events)

          lyric_events = [e for e in enhanced_single_track_score if e[0] == 'lyric']
          num_lyric_events = len(lyric_events)

          other_events = [e for e in enhanced_single_track_score if e[0] not in ['note', 'patch_change', 'text_event', 'lyric']]
          num_other_events = len(other_events)
          
          for c in cscore:
            tones = sorted(set([t[4] % 12 for t in c if t[3] != 9]))

            if tones:
              chords_tones.append(tones)
              all_tones.extend(tones)

              if tones not in ALL_CHORDS:
                all_chords_good = False
                bad_chords.append(tones)
                bad_chords_count += 1
          
          analysis['Number of ticks per quarter note'] = num_ticks
          analysis['Number of tracks'] = num_tracks
          analysis['Number of all events'] = len(enhanced_single_track_score)
          analysis['Number of patch change events'] = num_patch_changes
          analysis['Number of text events'] = num_text_events
          analysis['Number of lyric events'] = num_lyric_events
          analysis['All text and lyric events Latin'] = text_and_lyric_events_latin
          analysis['Number of other events'] = num_other_events
          analysis['Number of score notes'] = len(score_notes)
          analysis['Number of score chords'] = len(cscore)
          analysis['Score patches'] = sorted(set(score_patches))
          analysis['Score pitches'] = sorted(set(score_pitches))
          analysis['Score tones'] = sorted(set(all_tones))
          if chords_tones:
            analysis['Shortest chord'] = sorted(min(chords_tones, key=len))
            analysis['Longest chord'] = sorted(max(chords_tones, key=len))
          analysis['All chords good'] = all_chords_good
          analysis['Number of bad chords'] = bad_chords_count
          analysis['Bad chords'] = sorted([list(c) for c in set(tuple(bc) for bc in bad_chords)])

      else:
        analysis['Error'] = 'Provided score does not have specified patches to analyse'
        analysis['Provided patches to analyse'] = sorted(patches_to_analyze)
        analysis['Patches present in the score'] = sorted(set(all_score_patches))

      if return_enhanced_monophonic_melody:

        score_notes_copy = copy.deepcopy(score_notes)
        chordified_score_notes = chordify_score(score_notes_copy)

        melody = [c[0] for c in chordified_score_notes]

        fixed_melody = []

        for i in range(len(melody)-1):
          note = melody[i]
          nmt = melody[i+1][1]

          if note[1]+note[2] >= nmt:
            note_dur = nmt-note[1]-1
          else:
            note_dur = note[2]

          melody[i][2] = note_dur

          fixed_melody.append(melody[i])
        fixed_melody.append(melody[-1])

      if return_score_tones_chords:
        cscore = chordify_score(score_notes)
        for c in cscore:
          tones_chord = sorted(set([t[4] % 12 for t in c if t[3] != 9]))
          if tones_chord:
            tones_chords.append(tones_chord)

      if return_chordified_enhanced_score_with_lyrics:
        score_with_lyrics = [e for e in enhanced_single_track_score if e[0] in ['note', 'text_event', 'lyric']]
        chordified_enhanced_score_with_lyrics = chordify_score(score_with_lyrics)
      
      # Returned data

      requested_data = []

      if return_score_analysis and analysis:
        requested_data.append([[k, v] for k, v in analysis.items()])

      if return_enhanced_score and enhanced_single_track_score:
        requested_data.append([num_ticks, enhanced_single_track_score])

      if return_enhanced_score_notes and score_notes:
        requested_data.append(score_notes)

      if return_enhanced_monophonic_melody and fixed_melody:
        requested_data.append(fixed_melody)
        
      if return_chordified_enhanced_score and cescore:
        requested_data.append(cescore)

      if return_chordified_enhanced_score_with_lyrics and chordified_enhanced_score_with_lyrics:
        requested_data.append(chordified_enhanced_score_with_lyrics)

      if return_score_tones_chords and tones_chords:
        requested_data.append(tones_chords)

      if return_text_and_lyric_events and text_and_lyric_events:
        requested_data.append(text_and_lyric_events)

      return requested_data
  
  else:
    return ['Check score for errors and compatibility!']

###################################################################################

def load_signatures(signatures_data, covert_counts_to_ratios=False, omit_drums=True):

    sigs_dicts = []
    
    for sig in tqdm.tqdm(signatures_data):

        if omit_drums:
            sig = [sig[0], [s for s in sig[1] if s[0] < 449]]

        if covert_counts_to_ratios:
            tcount = sum([s[1] for s in sig[1]])
            sig = [sig[0], [[s[0], s[1] / tcount] for s in sig[1]]]
    
        sigs_dicts.append([sig[0], dict(sig[1])])

    return sigs_dicts

###################################################################################

def get_distance(sig_dict1, 
                 sig_dict2,
                 mismatch_penalty=10,
                 p=3
                ):

    all_keys = set(sig_dict1.keys()) | set(sig_dict2.keys())
    
    total = 0.0
    
    for key in all_keys:

        if key in sig_dict1 and key in sig_dict2:
            a = sig_dict1.get(key, 0)
            b = sig_dict2.get(key, 0)
    
            if min(a, b) > 0:
                ratio = max(a, b) / min(a, b)
                diff = ratio - 1
                total += diff ** p

        else:
            diff = mismatch_penalty
            total += diff ** p
            
    return total ** (1.0 / p)

###################################################################################

def get_distance_np(sig_dict1, sig_dict2, mismatch_penalty=10, p=3):

    keys = np.array(list(set(sig_dict1.keys()) | set(sig_dict2.keys())))

    freq1 = np.array([sig_dict1.get(k, 0) for k in keys], dtype=float)
    freq2 = np.array([sig_dict2.get(k, 0) for k in keys], dtype=float)
    
    mask = (freq1 > 0) & (freq2 > 0)
    
    diff = np.where(mask,
                    (np.maximum(freq1, freq2) / np.minimum(freq1, freq2)) - 1.0,
                    mismatch_penalty)
    
    sum_term = np.sum((diff ** p) * union_mask, axis=1)
    
    return np.cbrt(sum_term) if p == 3 else np.power(sum_term, 1.0 / p)

###################################################################################

def counter_to_vector(counter, union_keys):

    vec = np.zeros(union_keys.shape, dtype=float)
    keys   = np.array(list(counter.keys()))
    values = np.array(list(counter.values()), dtype=float)
    indices = np.searchsorted(union_keys, keys)
    vec[indices] = values
    
    return vec

###################################################################################

def precompute_signatures(signatures_dictionaries):

    all_counters = [sig[1] for sig in signatures_dictionaries]
    global_union = np.array(sorted({key for counter in all_counters for key in counter.keys()}))
    
    X = np.stack([counter_to_vector(sig[1], global_union) for sig in signatures_dictionaries])

    return X, global_union

###################################################################################

def get_distances_np(trg_signature_dictionary,
                    X,
                    global_union,
                    mismatch_penalty=10,
                    p=3
                    ):

    target_vec = counter_to_vector(trg_signature_dictionary, global_union)
    
    mask_both = (X > 0) & (target_vec > 0)
    
    diff = np.where(mask_both,
                    (np.maximum(X, target_vec) / np.minimum(X, target_vec)) - 1.0,
                    mismatch_penalty)
    
    union_mask = (X > 0) | (target_vec > 0)
    
    sum_term = np.sum((diff ** p) * union_mask, axis=1)
    
    return np.cbrt(sum_term) if p == 3 else np.power(sum_term, 1.0 / p)

###################################################################################

def get_MIDI_signature(path_to_MIDI_file,
                       transpose_factor=0,
                       covert_counts_to_ratios=False,
                       omit_drums=True
                      ):

    try:
    
        raw_score = midi2single_track_ms_score(path_to_MIDI_file)
        
        escore = advanced_score_processor(raw_score, return_enhanced_score_notes=True)[0]
        
        escore = augment_enhanced_score_notes(escore)
        
        drums_offset = len(ALL_CHORDS_SORTED) + 128
    
        transpose_factor = max(0, min(6, transpose_factor))
    
        if transpose_factor > 0:
            
            sidx = -transpose_factor
            eidx = transpose_factor
    
        else:
            sidx = 0
            eidx = 1
    
        src_sigs = []
        
        for i in range(sidx, eidx):
        
            escore_copy = copy.deepcopy(escore)
            
            for e in escore_copy:
                e[4] += i
            
            cscore = chordify_score([1000, escore_copy])
            
            sig = []
            dsig = []
            
            for c in cscore:
                
              all_pitches = [e[4] if e[3] != 9 else e[4]+128 for e in c]
              chord = sorted(set(all_pitches))
            
              pitches = sorted([p for p in chord if p < 128], reverse=True)
              drums = [(d+drums_offset)-128 for d in chord if d > 127]
            
              if pitches:
                if len(pitches) > 1:
                    
                  tones_chord = sorted(set([p % 12 for p in pitches]))
            
                  if tones_chord not in ALL_CHORDS_SORTED:
                      tones_chord = check_and_fix_tones_chord(tones_chord)
                      
                  sig_token = ALL_CHORDS_SORTED.index(tones_chord) + 128
            
                elif len(pitches) == 1:
                  sig_token = pitches[0]
            
                sig.append(sig_token)
            
              if drums:
                  dsig.extend(drums)
    
    
            if omit_drums:
                sig_p = dict.fromkeys(sig, 0)
                
                for item in sig:
                    sig_p[item] += 1
    
            else:
                sig_p = dict.fromkeys(sig+dsig, 0)
            
                for item in sig+dsig:
                    sig_p[item] += 1

            if covert_counts_to_ratios:
                tcount = sum([s[1] for s in sig_p.items()])
                sig_p = dict([[s[0], s[1] / tcount] for s in sig_p.items()])
            
            src_sigs.append(sig_p)
            
        return src_sigs
        
    except:
        return []   

###################################################################################

def load_pickle(input_file_name, ext='.pickle', verbose=True):

    if input_file_name:

        if verbose:
            print('Tegridy Pickle File Loader')
            print('Loading the pickle file. Please wait...')
        
        if os.path.basename(input_file_name).endswith(ext):
            fname = input_file_name
        
        else:
            fname = input_file_name + ext
        
        with open(fname, 'rb') as pickle_file:
            content = pickle.load(pickle_file)
        
        if verbose:
            print('Done!')
        
        return content

    else:
        return None

###################################################################################

def search_and_filter(sigs_dicts,
                      X,
                      global_union,
                      monster_dir = './Monster-MIDI-Dataset/MIDIs/',
                      master_dir = './Master-MIDI-Dataset/',
                      output_dir = './Output-MIDI-Dataset/',
                      number_of_top_matches_to_copy = 30,
                      transpose_factor=6
                     ):

    transpose_factor = max(0, min(6, transpose_factor))
    
    if transpose_factor > 0:
        
        tsidx = -transpose_factor
        teidx = transpose_factor
    
    else:
        tsidx = 0
        teidx = 1

    master_midis = create_files_list([master_dir])

    os.makedirs(master_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    for midi in master_midis:
    
        inp_fn = os.path.basename(midi)
    
        print('=' * 70)
        print('Processing MIDI file:', inp_fn)
        print('=' * 70)
    
        trg_sigs = get_MIDI_signature(midi, transpose_factor=transpose_factor)
    
        tv = list(range(tsidx, teidx))
        
        seen = []
        rseen = []
    
        for i in tqdm.tqdm(range(len(trg_sigs))):
            
            dists = get_distances_np(trg_sigs[i], X, global_union)
        
            sorted_indices = np.argsort(dists).tolist()
    
            out_dir = os.path.splitext(inp_fn)[0]
    
            os.makedirs(output_dir+'/'+out_dir, exist_ok=True)
        
            for _, idx in enumerate(sorted_indices[:number_of_top_matches_to_copy]):          
                
                fn = sigs_dicts[idx][0]
                dist = dists[idx]
        
                new_fn = output_dir+out_dir+'/'+str(dist)+'_'+str(tv[i])+'_'+fn+'.mid'
        
                if fn not in seen and dist not in rseen:
                    
                    src_fn = monster_dir+fn[0]+'/'+fn+'.mid'
                    
                    if os.path.exists(src_fn):
                        shutil.copy2(src_fn, new_fn)
                        seen.append(fn)
                        rseen.append(dist)

    print('=' * 70)
    print('Done!')
    print('=' * 70)

###################################################################################

print('Module is loaded!')
print('Enjoy! :)')
print('=' * 70)

###################################################################################
# This is the end of the monster_search_and_filter Python module
###################################################################################