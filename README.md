# NeuralSongGenerator
A generator that creates a song (lyrics and chords) and play it

This is the [demo](https://colab.research.google.com/drive/1ql2kEv5pAP955LcebTXBEOsJZz-l2SDF?usp=sharing) 

## Recipe for song generation
1. Parse song and lyrics with `BeautifulSoup4`
2. Fine-tune GPT2 with chords and lyrics
3. Read texts from GPT2 with [uberduck.ai](https://uberduck.ai/#voice=eminem)
4. Play generated chords with `PrettyMIDI` with midi files downloaded from [here](https://musical-artifacts.com/artifacts/)
5. You can overlap voice and chords with `pydub` library
6. Also you can generate an album cover with [DeepDaze](https://github.com/lucidrains/deep-daze)

Generated [example](https://drive.google.com/drive/folders/1CGHCU5CehsB-fv3nnfk05uV6udvsF5o9?usp=sharing)
