""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text import cmudict, pinyin
import os 

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_silences = ["@sp", "@spn", "@sil"]


if os.environ.get('PAPAREO_HACKS'):


    # based on looking at what actually occurs in PeterK and PeterL data 

    silences = ['sp',  'sil',  'spn']
    letters = ['a', 'e', 'i', 'o', 'u', 'm', 'r', 'p', 'w', 't', 'k', 'h', 'n', 'ŋ', 'f']
    long_vowels = ['a:', 'e:', 'o:', 'i:', 'u:']
    diphthongs = ['ai', 'au', 'ei',  'ae',  'oe', 'ao', 'oi', 'ou']
    long_diphthongs = ['a:u', 'a:i', 'a:o', 'a:e', 'o:u', 'o:i', 'e:i']
    # these should probably not occur??
    weird_ones = ['asp', 'ki', '0', 'ap', 'eoi', 'eu']

    # Export Māori symbols:
    symbols = (
        [_pad]
        + list(_special)
        + list(_punctuation)
        + silences
        + letters
        + long_vowels
        + diphthongs
        + long_diphthongs 
        + weird_ones
    )

else:
        
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
    _arpabet = ["@" + s for s in cmudict.valid_symbols]
    _pinyin = ["@" + s for s in pinyin.valid_symbols]

    # Export all symbols:
    symbols = (
        [_pad]
        + list(_special)
        + list(_punctuation)
        + list(_letters)
        + _arpabet
        + _pinyin
        + _silences
    )
