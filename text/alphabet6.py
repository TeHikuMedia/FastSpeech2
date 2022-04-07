'''
An alphabet derived from pairs of consonant+(vowel|long vowel|diphthong).
Diphthong is as defined by te reo Māori speakers.

A sample is provided to encode text to this alphabet. We take a hungry
approach where the longest string is prioritised. Sometimes prefixes
break this rule so we provide a mechanism to handle them as an edge
case.

This alphabet was inspired by the way we learnt soungs growing up. When
reading the polynesian alphabet, it's common to read the consonants as
"he, ke, la, mu, pu, nu, we,". When practicing diphthongs, it oftens
helps to practice them in context e.g. wai, kou, kau, tae.

- @Keoni 2022-04-04

'''

import os
from unidecode import unidecode
from random import random

CONSONANTS = ['h', 'k', 'm', 'n', 'p', 'r', 't', 'w', 'wh', 'ng']
VOWELS = ['a', 'e', 'i', 'o', 'u']
LONG_VOWELS = ['ā', 'ē', 'ī', 'ō', 'ū']
DIPHTHONGS = [
    'ae',
    'ai',
    'ao',
    'au',
    'ei',
    'oi',
    'oe',
    'ou',
    'āe',
    'āi',
    'āo',
    'āu',
    'ēi',
    'ōi',
    'ōe',
    'ōu',
]

def alphabet6():
    for mea in DIPHTHONGS + LONG_VOWELS + VOWELS:
        for consonant in CONSONANTS + [' ']:
            s = (consonant+mea).strip()
            yield s
