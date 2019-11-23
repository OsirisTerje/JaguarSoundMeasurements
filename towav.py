from os import path
from pydub import AudioSegment


# files                                                                         
src = ["EinarBil.mp3","EinarBilLF.mp3","TerjesBil.mp3","TerjeBilLF.mp3"]

# convert wav to mp3                                                            
for asrc in src:
    sound = AudioSegment.from_mp3(asrc)
    dst = path.splitext(asrc)[0]+'.wav'
    print ("Convert "+asrc+" to "+dst)
    sound.export(dst, format="wav")
print ("Converted all")