""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text import cmudict, pinyin, alphabet6
import os 

_pad = "_"
_punctuation = "!'\"(),.:;? "
_special = "-"

if os.environ.get('PAPAREO_HACKS'):

    alphabet6_occurences =  ['a', 'ae', 'ai', 'ao', 'au', 'e', 'ei', 'h', 'ha', 'hae', 'hai', 'hau', 'he', 'hei', 'hi', 'ho', 'hou', 'hu', 'hā', 'hāe', 'hāo', 'hē', 'hī', 'hō', 'hū', 'i', 'ka', 'kae', 'kai', 'kau', 'ke', 'kei', 'ki', 'ko', 'koe', 'koi', 'kou', 'ku', 'kā', 'kāi', 'kāo', 'kāu', 'kē', 'kī', 'kō', 'kōi', 'kū', 'm', 'ma', 'mae', 'mai', 'mao', 'mau', 'me', 'mi', 'mo', 'moe', 'mou', 'mu', 'mā', 'māi', 'māo', 'māu', 'mē', 'mī', 'mō', 'mōu', 'n', 'na', 'nai', 'nau', 'ne', 'nei', 'nga', 'ngai', 'ngao', 'ngau', 'nge', 'ngi', 'ngo', 'ngoi', 'ngou', 'ngu', 'ngā', 'ngū', 'ni', 'no', 'noi', 'nu', 'nā', 'nāi', 'nāu', 'nē', 'nō', 'nōu', 'o', 'oe', 'oi', 'ou', 'pa', 'pae', 'pai', 'pao', 'pau', 'pe', 'pei', 'phone', 'pi', 'po', 'poi', 'pou', 'pu', 'pā', 'pāe', 'pāi', 'pāo', 'pē', 'pī', 'pō', 'pōu', 'pū', 'r', 'ra', 'rae', 'rai', 'rao', 'rau', 're', 'rei', 'ri', 'ro', 'roi', 'rou', 'ru', 'rā', 'rāi', 'rāo', 'rāu', 'rē', 'rī', 'rō', 'rōi', 'rū', 'sp', 't', 'ta', 'tae', 'tai', 'tao', 'tau', 'te', 'tei', 'ti', 'to', 'toe', 'toi', 'tou', 'tu', 'tā', 'tāi', 'tāo', 'tāu', 'tē', 'tēi', 'tī', 'tō', 'tōu', 'tū', 'u', 'wa', 'wae', 'wai', 'we', 'wha', 'whae', 'whai', 'whe', 'whi', 'whu', 'whā', 'whāi', 'whāo', 'whī', 'whū', 'wi', 'wou', 'wā', 'wāi', 'wī', 'ā', 'āe', 'āi', 'āu', 'ē', 'ī', 'ō', 'ōe', 'ōu', 'ū',]
    # Export alphabet6 Māori syllabless:
    symbols = (
        [_pad]
        + list(_special)
        + list(_punctuation)
        + alphabet6_occurences # use this for slight efficiency gain
        #+ list(alphabet6())   # instead of this 
    )

else:
        
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
    _arpabet = ["@" + s for s in cmudict.valid_symbols]
    _pinyin = ["@" + s for s in pinyin.valid_symbols]
    _silences = ["@sp", "@spn", "@sil"]

    # Export all symbols:
    symbols = (
        [_pad]
        + list(_special)
        + list(_punctuation)
        + list(letters)
        + _arpabet
        + _pinyin
        + _silences
    )
