# Author: Martin Konečnik
# Contact: martin.konecnik@gmail.com
# Licenced under MIT

# 4- Lastnosti Fourirjeve Analize
# Potrebna ročna inštalacija knjižnice pytftb (najlažje z git): https://github.com/scikit-signal/pytftb
import numpy as np
from scipy.signal import convolve2d as conv2
from scipy.signal.windows import gaussian as gausswin
from scipy.signal import hilbert
from tftb.generators import fmlin
from tftb.processing import Spectrogram
from tftb.processing.reassigned import pseudo_wigner_ville
from tftb.processing import WignerVilleDistribution
import pylab as pylab
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Potrebni za del z zvokom
import sounddevice as sd
from scipy.io import wavfile
import time

# ukazi.m:1
# --------------------------------------------------------------------------------------
# Časovno-frekvenčna ločljivost in časovno-frekvenčne porazdelitve
#######################################################################

# ukazi.m:6 -- Note: Preverjeno z Matlab
Fs = 1024
sig = fmlin(1*Fs)[0]    # Ker v pythonu fmlin vrne analitičen signal, ni potrebno napraviti Hilbertove transformacije.

# Narišemo realni del v časovni domeni (signal je kompleksen, a le zaradi umetno zagotovljene analitičnosti.
plt.figure()
plt.plot(np.real(sig), LineWidth=0.4)
plt.xlabel('t')
plt.ylabel('amplituda')

# Analitični signali nimajo simetrije preko Nyquistove frekvence.
plt.figure()
plt.plot(abs(np.fft.fft(sig)), LineWidth=0.4)
plt.xlabel('f')
plt.ylabel('amplituda')

# ukazi.m:16
#######################################################################
# Časovno-frekvenčna ločljivost prvič: višja časovna ločljivost, nižja frekvenčna ločljivost

# ukazi.m:18 -- Note: Graf je prikazan s knjižnico pytftb in se razlikuje od Matlabovega.
plt.close('all')
T=32
N=256
dT=T - 1
window = np.ones(T)
TFD = Spectrogram(sig, n_fbins=N, fwindow=window)
TFD.run()
TFD.plot(kind="contour", threshold=0.1, show_tf=False)

# plt.xlabel('t (s)','FontSize',12)
# plt.ylabel('f (Hz)','FontSize',12)
# plt.title('T={0},N={1},dt={2}'.format(T, N, dT))

# ukazi.m:29 -- Note: Isto kot prej.
# Časovno-frekvenčna ločljivost drugič: nižja časovna ločljivost, višja frekvenčna ločljivost
T=256
N=256
dT=T - 1
window = np.ones(T)
TFD = Spectrogram(sig, n_fbins=N, fwindow=window)
TFD.run()
TFD.plot(kind="contour", threshold=0.1, show_tf=False)

# plt.xlabel('t (s)','FontSize',12)
# plt.ylabel('f (Hz)','FontSize',12)
# plt.title(cat('T=',str(T),',N=',str(N),',dT=',str(dT)))

# ukazi.m:49 -- Note:
# Wigner-Villova časovno-frekvenčna porazdelitev - skoraj idealna časovna in frekvenčna ločljivost

wvd = WignerVilleDistribution(np.real(sig))
wvd.run()
wvd.plot(kind='contour')

tfr, rtfr, hat = pseudo_wigner_ville(np.real(sig))

TFD,t,f=tfrwv(sig,nargout=3)
# ukazi4.m:41
plt.figure()
imagesc(t,f,TFD)
plt.axis('tight')
plt.xlabel('t','FontSize',12)
plt.ylabel('f','FontSize',12)
plt.axis('tight')
plt.axis('xy')
plt.title(cat('Wigner-Villova asovno-frekvenna porazdelitev'))

# Trenutna autokorelacija  Rss(t,tau) = sig(t+tau) .* conj(sig(t-tau)); -  kvadratna funkcija signala---------------------
#   je osnova za Wigner-Villovo asovnofrekvenno porazdelitev in omogoa skoraj idealno asovno in frekvenno loljivost
t=np.arange(1,size(sig))
# ukazi4.m:51

tcol=size(sig)
# ukazi4.m:52

Rss=np.zeros(tcol,tcol)
# ukazi4.m:53

for icol in np.arange(1,tcol).reshape(-1):
    ti=t[icol]
# ukazi4.m:55
    taumax=min(cat(ti - 1,tcol - ti,round(tcol / 2) - 1))
# ukazi4.m:56
    tau=np.arange(- taumax,taumax)
# ukazi4.m:58
    indices=rem(tcol + tau,tcol) + 1
# ukazi4.m:59
    Rss[indices,icol]=multiply(sig[ti + tau],conj(sig[ti - tau]))
# ukazi4.m:60

plt.figure()

for icol in np.arange(1,tcol / 2,10).reshape(-1):
    cla
    plt.subplot(2,1,1)
    plt.plot(np.real(Rss[:,icol]))
    plt.title(cat('t = ',str(icol),', np.real Rss'))
    plt.axis('tight')
    plt.subplot(2,1,2)
    plt.plot(imag(Rss[:,icol]))
    plt.title(cat('t = ',str(icol),', imag Rss'))
    plt.axis('tight')
    plt.waitforbuttonpress()

# Wigner-Villova asovno-frekvenna porazdelitev - skoraj idealna asovna in frekvenna loljivost----------------------------
# Wigner-Villova asovno-frekvenna porazdelitev je Fourierova transformacija trenutne prene korelacije Rss.
TFD,t,f=tfrwv(sig,nargout=3)
# ukazi4.m:73

plt.figure()
contour(t,f,TFD)
plt.axis('tight')
plt.xlabel('t','FontSize',12)
plt.ylabel('f','FontSize',12)
plt.axis('tight')
plt.title(cat('Wigner-Villova asovno-frekvenna porazdelitev'))
'------------------------------------------------------------------------------------------------------------------------------'
# asovno-frekvenna loljivost in asovno-frekvenne porazdelitve - ve signalov skupaj
# Primer 1: odkomentirajte za zgled z dvema asimetrino razporejenima atomoma v asovno-frekvenni ravnini.
Fs=1024
# ukazi4.m:83
sig=atoms(1*Fs,cat([1*Fs / 4,0.15,20,1],[3*Fs / 4,0.35,20,1]))
# ukazi4.m:84


plt.figure()
plt.plot(np.real(sig))

plt.xlabel('t')
plt.ylabel('amplituda')
plt.axis('tight')
plt.figure()
plt.plot(abs(np.fft.fft(sig)))

plt.xlabel('f')
plt.ylabel('amplituda')
plt.axis('tight')
#######################################################################
# asovno-frekvenna loljivost prvi: vija asovna loljivost, nija frekvenna loljivost---------------------------------
plt.close('all')
T=32
# ukazi4.m:99
N=256
# ukazi4.m:100
dT=T - 1
# ukazi4.m:101
TFD,f,t=specgram(sig,N,Fs,window(rectwin,T),dT,nargout=3)
# ukazi4.m:102
plt.figure()
contour(t,f,abs(TFD))
plt.axis('tight')
plt.xlabel('t (s)','FontSize',12)
plt.ylabel('f (Hz)','FontSize',12)
plt.axis('tight')
plt.title(cat('Spektrogram: T=',str(T),',N=',str(N),',dT=',str(dT)))
# asovno-frekvenna loljivost drugi: nija asovna loljivost, vija frekvenna loljivost--------------------------------
T=256
# ukazi4.m:110
N=256
# ukazi4.m:111
dT=T - 1
# ukazi4.m:112
TFD,f,t=specgram(sig,N,Fs,window(rectwin,T),dT,nargout=3)
# ukazi4.m:113
plt.figure()
contour(t,f,abs(TFD))
plt.axis('tight')
plt.xlabel('t (s)','FontSize',12)
plt.ylabel('f (Hz)','FontSize',12)
plt.axis('tight')
plt.title(cat('Spektrogram: T=',str(T),',N=',str(N),',dT=',str(dT)))
# Wigner-Villova asovno-frekvenna porazdelitev - skoraj idealna asovna in frekvenna loljivost----------------------------
TFD,t,f=tfrwv(sig,nargout=3)
# ukazi4.m:121
plt.figure()
contour(t,f,abs(TFD),100)
plt.axis('tight')
plt.xlabel('t','FontSize',12)
plt.ylabel('f','FontSize',12)
plt.axis('tight')
plt.title(cat('Wigner-Villova asovno-frekvenna porazdelitev'))
# Wigner-Villova asovno-frekvenna porazdelitev - SAMO np.realNI DEL SIGNALA - Zrcaljenje preko frekce Fsamp/4 !!! ---------------------------
TFD,t,f=tfrwv(np.real(sig),nargout=3)
# ukazi4.m:129
plt.figure()
contour(t,f,abs(TFD),100)
plt.axis('tight')
plt.xlabel('t','FontSize',12)
plt.ylabel('f','FontSize',12)
plt.axis('tight')
plt.title(cat('Wigner-Villova asovno-frekvenna porazdelitev'))
# pseudo-Wigner-Villova asovno-frekvenna porazdelitev -----------------------------------------------------------------------
#    - okno v asovni domeni, ki prepreuje interference med asovno odmaknjenimi atomi
TFD,t,f=tfrpwv(sig,np.arange(1,Fs),Fs,hamming(31),nargout=3)
# ukazi4.m:139
plt.figure()
contour(t,f,abs(TFD))
plt.axis('tight')
plt.xlabel('t','FontSize',12)
plt.ylabel('f','FontSize',12)
plt.axis('tight')
plt.title(cat('Psevdo Wigner-Villova asovno-frekvenna porazdelitev'))
# pseudo-Wigner-Villova asovno-frekvenna porazdelitev - SAMO np.realNI DEL SIGNALA - Zrcaljenje preko frekce Fsamp/4  !!! -----------------------------------------
#    - okno v asovni domeni, ki prepreuje interference med asovno odmaknjenimi atomi
TFD,t,f=tfrpwv(np.real(sig),np.arange(1,Fs),Fs,hamming(31),nargout=3)
# ukazi4.m:148
plt.figure()
contour(t,f,abs(TFD))
plt.axis('tight')
plt.xlabel('t','FontSize',12)
plt.ylabel('f','FontSize',12)
plt.axis('tight')
plt.title(cat('Psevdo Wigner-Villova asovno-frekvenna porazdelitev'))
'------------------------------------------------------------------------------------------------------------------------------'
# asovno-frekvenna loljivost in asovno-frekvenne porazdelitve - ve signalov skupaj

# # Primer 2: zgled z tirimi simetrino razporejenimi atomi v asovno-frekvenni ravnini.
Fs=1024
# ukazi4.m:161
sig=atoms(1*Fs,cat([1*Fs / 4,0.15,20,1],[3*Fs / 4,0.15,20,1],[1*Fs / 4,0.35,20,1],[3*Fs / 4,0.35,20,1]))
# ukazi4.m:162
plt.figure()
plt.plot(np.real(sig))

plt.xlabel('t')
plt.ylabel('amplituda')
plt.axis('tight')
plt.figure()
plt.plot(abs(np.fft.fft(sig)))

plt.xlabel('f')
plt.ylabel('amplituda')
plt.axis('tight')
#######################################################################
# asovno-frekvenna loljivost prvi: vija asovna loljivost, nija frekvenna loljivost---------------------------------
plt.close('all')
T=32
# ukazi4.m:178
N=256
# ukazi4.m:179
dT=T - 1
# ukazi4.m:180
TFD,f,t=specgram(sig,N,Fs,window(rectwin,T),dT,nargout=3)
# ukazi4.m:181
plt.figure()
imagesc(t,f[1:end() / 2],abs(TFD[1:end() / 2,:]))
plt.axis('xy')
plt.xlabel('t (s)','FontSize',12)
plt.ylabel('f (Hz)','FontSize',12)
plt.title(cat('Spektrogram: T=',str(T),',N=',str(N),',dT=',str(dT)))

T=256
# ukazi4.m:188
N=256
# ukazi4.m:189
dT=T - 1
# ukazi4.m:190
TFD,f,t=specgram(sig,N,Fs,window(rectwin,T),dT,nargout=3)
# ukazi4.m:191
plt.figure()
imagesc(t,f[1:end() / 2],abs(TFD[1:end() / 2,:]))
plt.axis('xy')
plt.xlabel('t (s)','FontSize',12)
plt.ylabel('f (Hz)','FontSize',12)
plt.axis('tight')
plt.title(cat('Spektrogram: T=',str(T),',N=',str(N),',dT=',str(dT)))
# Wigner-Villova asovno-frekvenna porazdelitev - skoraj idealna asovna in frekvenna loljivost----------------------------
TFD,t,f=tfrwv(sig,nargout=3)
# ukazi4.m:199
plt.figure()
contour(t,f,abs(TFD))
plt.axis('xy')
plt.xlabel('t','FontSize',12)
plt.ylabel('f','FontSize',12)
plt.axis('tight')
plt.title(cat('Wigner-Villova asovno-frekvenna porazdelitev'))
# pseudo-Wigner-Villova asovno-frekvenna porazdelitev -----------------------------------------------------------------------
#    - okno v asovni domeni, ki prepreuje interference med asovno odmaknjenimi atomi
TFD,t,f=tfrpwv(sig,np.arange(1,Fs),Fs,hamming(31),nargout=3)
# ukazi4.m:208
plt.figure()
contour(t,f,abs(TFD))
plt.axis('xy')
plt.xlabel('t','FontSize',12)
plt.ylabel('f','FontSize',12)
plt.axis('tight')
plt.title(cat('Psevdo Wigner-Villova asovno-frekvenna porazdelitev'))
# pseudo-Wigner-Villova asovno-frekvenna porazdelitev - SAMO np.realNI DEL SIGNALA - Zrcaljenje preko frekce Fsamp/4 !!!! ----------------------------------------
#    - okno v asovni domeni, ki prepreuje interference med asovno odmaknjenimi atomi
TFD,t,f=tfrpwv(np.real(sig),np.arange(1,Fs),Fs,hamming(31),nargout=3)
# ukazi4.m:217
plt.figure()
contour(t,f,abs(TFD))
plt.axis('xy')
plt.xlabel('t','FontSize',12)
plt.ylabel('f','FontSize',12)
plt.axis('tight')
plt.title(cat('Psevdo Wigner-Villova asovno-frekvenna porazdelitev'))
# Choi-Williams asovno-frekvenna porazdelitev ------------------------------------------------------------------------------
#      - okno v asovni domeni, ki prepreuje interference med asovno odmaknjenimi atomi
#      - okno v frekvenni domeni, ki prepreuje interference med frekvenno odmaknjenimi atomi
TFD,t,f=tfrcw(sig,np.arange(1,Fs),Fs,hamming(31),hamming(31),nargout=3)
# ukazi4.m:228
plt.figure()
imagesc(t,f,abs(TFD))
plt.axis('xy')
plt.xlabel('t','FontSize',12)
plt.ylabel('f','FontSize',12)
plt.axis('tight')
plt.title(cat('Choi-Williams asovno-frekvenna porazdelitev'))
'-----------------------------------------------------------------------------------------------'
# asovno-frekvenne porazdelitve: primeri za igranje in uenje
#######################################################################################5
T=1
# ukazi4.m:238

Fs=512
# ukazi4.m:239

# Primer 1: analitien signal z linearno naraajoo frekvenco.
sig,trenutnaFrekvenca=fmlin(dot(T,Fs),nargout=2)
# ukazi4.m:242
sig=scipy.signal.hilbert(sig)
# ukazi4.m:243

# Primer 2: analitien signal s sinusoidno spremeinjajoo se frekvenco.
sig,trenutnaFrekvenca=fmsin(dot(T,Fs),0.05,0.45,100,20,0.3,- 1.0,nargout=2)
# ukazi4.m:246
sig=scipy.signal.hilbert(sig)
# ukazi4.m:247

# Primer 3: signal s eksponentno spremeinjajoo se frekvenco.
sig,trenutnaFrekvenca=fmpower(dot(T,Fs),0.5,cat(1,0.5),cat(100,0.1),nargout=2)
# ukazi4.m:250
sig=scipy.signal.hilbert(sig)
# ukazi4.m:251

# Od tu dalje je je koda za katerikoli zgornji primer.
plt.figure()
plt.subplot(2,1,1)

plt.plot(cat(np.arange(0,size(sig) - 1)) / Fs,np.real(sig))
plt.axis('tight')
plt.xlabel('t (s)')
plt.ylabel('amplituda')
plt.title('signal')

plt.subplot(2,1,2)

plt.plot(cat(np.arange(0,size(sig) - 1)) / Fs,dot(trenutnaFrekvenca,Fs))
plt.xlabel('t (s)')
plt.ylabel('f (Hz)')
plt.title('trenutna frekvenca')
plt.axis('tight')
# asovno-frekvenna loljivost prvi: vija asovna loljivost, nija frekvenna loljivost---------------------------------
plt.close('all')
winT=32
# ukazi4.m:263
N=256
# ukazi4.m:264
dT=winT - 1
# ukazi4.m:265
TFD,f,t=specgram(sig,N,Fs,window(rectwin,winT),dT,nargout=3)
# ukazi4.m:266
plt.figure()
contour(t,f[1:end() / 2],abs(TFD[1:end() / 2,:]))
plt.axis('xy')
plt.xlabel('t (s)','FontSize',12)
plt.ylabel('f (Hz)','FontSize',12)
plt.title(cat('Spektrogram: winT=',str(winT),',N=',str(N),',dT=',str(dT)))
# asovno-frekvenna loljivost drugi: nija asovna loljivost, vija frekvenna loljivost--------------------------------
winT=256
# ukazi4.m:273
N=256
# ukazi4.m:274
dT=winT - 1
# ukazi4.m:275
TFD,f,t=specgram(sig,N,Fs,window(rectwin,winT),dT,nargout=3)
# ukazi4.m:276
plt.figure()
imagesc(t,f[1:end() / 2],abs(TFD[1:end() / 2,:]))
plt.axis('xy')
plt.xlabel('t (s)','FontSize',12)
plt.ylabel('f (Hz)','FontSize',12)
plt.axis('tight')
plt.title(cat('Spektrogram: winT=',str(winT),',N=',str(N),',dT=',str(dT)))
# Wigner-Villova asovno-frekvenna porazdelitev - skoraj idealna asovna in frekvenna loljivost----------------------------
TFD,t,f=tfrwv(sig,np.arange(1,dot(T,Fs)),Fs,nargout=3)
# ukazi4.m:284
plt.figure()
contour(t,f,abs(TFD))
plt.axis('xy')
plt.xlabel('t','FontSize',12)
plt.ylabel('f','FontSize',12)
plt.axis('tight')
plt.title(cat('Wigner-Villova asovno-frekvenna porazdelitev'))
# pseudo-Wigner-Villova asovno-frekvenna porazdelitev -----------------------------------------------------------------------
#    - okno v asovni domeni, ki prepreuje interference med asovno odmaknjenimi atomi
TFD,t,f=tfrpwv(sig,np.arange(1,dot(T,Fs)),Fs,hamming(31),nargout=3)
# ukazi4.m:293
plt.figure()
contour(t,f,abs(TFD))
plt.axis('xy')
plt.xlabel('t','FontSize',12)
plt.ylabel('f','FontSize',12)
plt.axis('tight')
plt.title(cat('Psevdo Wigner-Villova asovno-frekvenna porazdelitev'))
# Choi-Williams asovno-frekvenna porazdelitev ------------------------------------------------------------------------------
#      - okno v asovni domeni, ki prepreuje interference med asovno odmaknjenimi atomi
#      - okno v frekvenni domeni, ki prepreuje interference med frekvenno odmaknjenimi atomi
TFD,t,f=tfrcw(sig,np.arange(1,dot(T,Fs)),Fs,hamming(25),hamming(25),nargout=3)
# ukazi4.m:303
plt.figure()
contour(t,f,abs(TFD))
plt.axis('xy')
plt.xlabel('t','FontSize',12)
plt.ylabel('f','FontSize',12)
plt.axis('tight')
plt.title(cat('Choi-Williams asovno-frekvenna porazdelitev'))
'------------------------------------------------------------------------------------------'
################################################################
# Primeri iz np.realnega ivljenja
print("\n" * 80)  # clc
# exit()
plt.close('all')
Fs=8000
# ukazi4.m:316

bits=8
# ukazi4.m:317

T=2
# ukazi4.m:318

#my_recorder=audiorecorder(Fs,bits,1)
# ukazi4.m:320
print('Snemam posnetek v dolini ',str(T),'s...'))
#recordblocking(my_recorder,T)
sig=sd.rec(T*Fs, Fs, 1)
# ukazi4.m:324
plt.figure()
plt.plot(cat(np.arange(1,dot(T,Fs))) / Fs,sig)
plt.title('Posneti signal')
plt.xlabel('t (s)')
plt.ylabel('amplituda')
wavplay(sig,Fs,'sync')
sig=scipy.signal.hilbert(sig)
# ukazi4.m:328

# od tu dalje je pa dodajte kodo sami. Govorimo seveda o asovno-frekvenni analizi.
# POZOR! asovno-frekvenne porazdelitve so lahko raunsko precej precej porene.
# Priporoam majhne korake pri poveavi doline signala in smelo nastavljanje tevila frekvennih tok.