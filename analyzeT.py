import pydub 
from pydub.silence import split_on_silence
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from numpy import fft as fft
from numpy import array


def plotit(data,time,plotnr):
    plt.subplot(plotnr)
    plt.plot(time, data, linewidth=0.02, alpha=0.8, color='k')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

def plotTimeDomain(time,channel1,channel2):
    #plot amplitude (or loudness) over time
    plt.figure(1,figsize=(16,12))
    plotit(channel1,time,211)
    plotit(channel2,time,212)
    plt.show()        

def printinfo(afile):
    print("Sound length "+str(len(afile)))
    print ("Frame rate "+str(afile.frame_rate))
    print ("Sample width "+str(afile.sample_width))
    print("Channels "+str(afile.channels))

def plotFFT(freqArray,fourier,filename,idx):
    magn = 10*np.log10(abs(fourier))
    start = 1000
    stop = idx*10
    plt.plot(freqArray[start:stop]/1000, magn[start:stop] , color='k', linewidth=0.02)
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Power (dB)')
    plt.savefig(filename+".png")
    plt.show()
    

def calcFFT(channel,rate,frate):
    fourier=fft.fft(channel)
    n = int(np.abs(len(channel)/2))
    #print("Length of channel1 "+str(n))
    fourier = fourier[0:n]
    # scale by the number of points so that the magnitude does not depend on the length
    fourier = fourier / float(n)
    #print("Length of final fourier array "+str(len(fourier)))
    maxValue = max(abs(fourier))
    #calculate the frequency at each point in Hz
    freqArray = np.arange(0, (n), 1.0) * (rate*2.0/n)
    freqs = np.fft.fftfreq(len(fourier))
    idx = np.argmax(np.abs(fourier))
    maxFreq = freqs[idx]*frate
    maxVal = abs(fourier[idx])
    # print ("Length of freqArray "+str(len(freqArray)))
    #print ("Max Value "+str('%.2f'%maxValue)+ " at "+str('%.2f'%maxFreq)+" Hz")
    #print ("Max Value 2 "+str(maxVal))
    return fourier,freqArray,idx,maxFreq,maxVal,

i=0

def process(file,result,freqa,vala,res,i):
    sound = pydub.AudioSegment.from_file(file+".wav",format="wav")
    #printinfo(sound)
    # From http://myinspirationinformation.com/uncategorized/audio-signals-in-python/
    #read wav file
    rate,audData=scipy.io.wavfile.read(file+".wav")

    channel1=audData[:,0] #left
    # channel2=audData[:,1] #right
    # time = np.arange(0, float(audData.shape[0]), 1) / rate
    # plotTimeDomain(time,channel1,channel2)

    # FFT
    fourier,freqArray,idx,maxfr,maxval = calcFFT(channel1,rate,sound.frame_rate)
    # plotFFT(freqArray,fourier,file,idx)
    level = 10*np.log10(fourier)
    maxv = max(level.real)
    # smaxv = str('%.2f'%maxv)
    # smaxfr = str('%.1f'%maxfr)
    #print ("Max db value:  "+smaxv+ "dB at "+smaxfr+" Hz")
    #res[i,0]=maxfr
    #res[i,1]=maxv
    res[i] = [maxfr, maxv]
    i += 1
    result.append((file,maxfr,maxv))
    freqa.append(maxfr)
    vala.append(maxv)
    return i


def isabel_plot(res1, res2, name_of_graph1, name_of_graph2,nameOfResult):
    #defines the figure
    plt.figure(1, figsize=(12,10))
    plt.suptitle('Relative sine wave noise injection Jaguar I-Pace')
    #removes unnecessary axes
    ax = plt.subplot(111)    
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)  
     
    plt.plot(res1[:,0], res1[:,1], c='b', label=name_of_graph1)
    plt.plot(res2[:,0], res2[:,1], c='r', label=name_of_graph2)
    plt.grid(color='lightgrey', ls = '--')
    
    #fixing the y-axis
    plt.ylim(0,50) #should work as set_ylim and set_ybound in one
    plt.yticks(range(0,51,1), [str(x)  if x%10 == 0 else ' ' for x in range(0,51,1)]) #defines ticks, with label at only each 10th
    
    #fixing the x-axis
    plt.xlim(800, 2600)
    plt.xticks(range(800,2601,200))
               
    #fixing the labels
    plt.xlabel('Frequency')
    plt.ylabel('dB', rotation='horizontal', position=(0.5,0.5))
    
    plt.legend() #do nothing here
    plt.savefig(nameOfResult,format='png')
    plt.show()    

i=0
res =np.zeros((10,2))




result = []
freq = []
val = []

i=process("TerjesBil-1K0",result,freq,val,res,i)
i=process("TerjesBil-1K2",result,freq,val,res,i)
i=process("TerjesBil-1K4",result,freq,val,res,i)
i=process("TerjesBil-1K6",result,freq,val,res,i)
i=process("TerjesBil-1K8",result,freq,val,res,i)
i=process("TerjesBil-2K0",result,freq,val,res,i)
i=process("TerjesBil-2K2",result,freq,val,res,i)
i=process("TerjesBil-2K4",result,freq,val,res,i)
i=process("TerjesBil-2K6",result,freq,val,res,i)
i=process("TerjesBil-2K8",result,freq,val,res,i)
#process("TerjesBil-3K0",result,freq,val)

print (res)
print(*result, sep='\n')
#f, ax = plt.subplots(1)
#ax.plot(res[:,0],res[:,1],color='b')

result = []
freq = []
val = []
i=0
res2 =np.zeros((10,2))



i=process("EinarBil-1K0",result,freq,val,res2,i)
i=process("EinarBil-1K2",result,freq,val,res2,i)
i=process("EinarBil-1K4",result,freq,val,res2,i)
i=process("EinarBil-1K6",result,freq,val,res2,i)
i=process("EinarBil-1K8",result,freq,val,res2,i)
i=process("EinarBil-2K0",result,freq,val,res2,i)
i=process("EinarBil-2K2",result,freq,val,res2,i)
i=process("EinarBil-2K4",result,freq,val,res2,i)
i=process("EinarBil-2K6",result,freq,val,res2,i)
i=process("EinarBil-2K8",result,freq,val,res2,i)

isabel_plot(res,res2,"Glass","Alum",'result.png')





    
