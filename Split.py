import pydub 
from pydub.silence import split_on_silence
import matplotlib.pyplot as plt
import numpy as np

sound = pydub.AudioSegment.from_file("EinarBil.wav",format="wav")
print("Sound length "+str(len(sound)))
chunks = split_on_silence(sound,1000,-40)
print ("Chunk length " + str(len(chunks)))

# FFT

bins,vals = chunks[1][1:3000].fft()
vals_normed = np.abs(vals)/len(vals)
plt.plot(bins/1000,vals_normed)
plt.show()

