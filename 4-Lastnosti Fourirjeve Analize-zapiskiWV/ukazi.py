# Autogenerated with SMOP 0.32-7-gcce8558
from smop.core import *
from matplotlib.pyplot import *
from sounddevice import *
from numpy import *
import os
import scipy.signal
from tftb.generators import fmlin
import sounddevice as sd
# 

'--------------------------------------------------------------------------------------'
# asovno-frekvenna loljivost in asovno-frekvenne porazdelitve
#######################################################################
#os.chdir('D:/Dropbox/FERI/mag/ROSIS/4-Lastnosti Fourirjeve Analize-zapiskiWV/tftb/tftb-0.2/tftb-0.2/mfiles')

Fs=1024
# ukazi4.m:6
sig=scipy.signal.hilbert(fmlin(1*Fs))
# ukazi4.m:7

figure()
plot(real(sig))

xlabel('t')
ylabel('amplituda')
axis('tight')
figure()
plot(abs(fft.fft(sig)))

xlabel('f')
ylabel('amplituda')
axis('tight')
#######################################################################
# asovno-frekvenna loljivost prvi: vija asovna loljivost, nija frekvenna loljivost
close('all')
T=32
# ukazi4.m:19
N=256
# ukazi4.m:20
dT=T - 1
# ukazi4.m:21
TFD,f,t=specgram(sig,N,Fs,window(rectwin,T),dT,nargout=3)
# ukazi4.m:22
figure()
contour(t,f,abs(TFD))
axis('tight')
xlabel('t (s)','FontSize',12)
ylabel('f (Hz)','FontSize',12)
axis('tight')
title(cat('T=',str(T),',N=',str(N),',dT=',str(dT)))
# asovno-frekvenna loljivost drugi: nija asovna loljivost, vija frekvenna loljivost
T=256
# ukazi4.m:30
N=256
# ukazi4.m:31
dT=T - 1
# ukazi4.m:32
TFD,f,t=specgram(sig,N,Fs,window(rectwin,T),dT,nargout=3)
# ukazi4.m:33
figure()
contour(t,f,abs(TFD))
axis('tight')
xlabel('t (s)','FontSize',12)
ylabel('f (Hz)','FontSize',12)
axis('tight')
title(cat('T=',str(T),',N=',str(N),',dT=',str(dT)))
# Wigner-Villova asovno-frekvenna porazdelitev - skoraj idealna asovna in frekvenna loljivost
TFD,t,f=tfrwv(sig,nargout=3)
# ukazi4.m:41
figure()
imagesc(t,f,TFD)
axis('tight')
xlabel('t','FontSize',12)
ylabel('f','FontSize',12)
axis('tight')
axis('xy')
title(cat('Wigner-Villova asovno-frekvenna porazdelitev'))
# Trenutna autokorelacija  Rss(t,tau) = sig(t+tau) .* conj(sig(t-tau)); -  kdavdratna funkcija signala---------------------
#   je osnova za Wigner-Villovo asovnofrekvenno porazdelitev in omogoa skoraj idealno asovno in frekvenno loljivost
t=arange(1,length(sig))
# ukazi4.m:51

tcol=length(sig)
# ukazi4.m:52

Rss=zeros(tcol,tcol)
# ukazi4.m:53

for icol in arange(1,tcol).reshape(-1):
    ti=t[icol]
# ukazi4.m:55
    taumax=min(cat(ti - 1,tcol - ti,round(tcol / 2) - 1))
# ukazi4.m:56
    tau=arange(- taumax,taumax)
# ukazi4.m:58
    indices=rem(tcol + tau,tcol) + 1
# ukazi4.m:59
    Rss[indices,icol]=multiply(sig[ti + tau],conj(sig[ti - tau]))
# ukazi4.m:60

figure()

for icol in arange(1,tcol / 2,10).reshape(-1):
    cla
    subplot(2,1,1)
    plot(real(Rss[:,icol]))
    title(cat('t = ',str(icol),', real Rss'))
    axis('tight')
    subplot(2,1,2)
    plot(imag(Rss[:,icol]))
    title(cat('t = ',str(icol),', imag Rss'))
    axis('tight')
    pause

# Wigner-Villova asovno-frekvenna porazdelitev - skoraj idealna asovna in frekvenna loljivost----------------------------
# Wigner-Villova asovno-frekvenna porazdelitev je Fourierova transformacija trenutne prene korelacije Rss.
TFD,t,f=tfrwv(sig,nargout=3)
# ukazi4.m:73

figure()
contour(t,f,TFD)
axis('tight')
xlabel('t','FontSize',12)
ylabel('f','FontSize',12)
axis('tight')
title(cat('Wigner-Villova asovno-frekvenna porazdelitev'))
'------------------------------------------------------------------------------------------------------------------------------'
# asovno-frekvenna loljivost in asovno-frekvenne porazdelitve - ve signalov skupaj
# Primer 1: odkomentirajte za zgled z dvema asimetrino razporejenima atomoma v asovno-frekvenni ravnini.
Fs=1024
# ukazi4.m:83
sig=atoms(1*Fs,cat([1*Fs / 4,0.15,20,1],[3*Fs / 4,0.35,20,1]))
# ukazi4.m:84


figure()
plot(real(sig))

xlabel('t')
ylabel('amplituda')
axis('tight')
figure()
plot(abs(fft.fft(sig)))

xlabel('f')
ylabel('amplituda')
axis('tight')
#######################################################################
# asovno-frekvenna loljivost prvi: vija asovna loljivost, nija frekvenna loljivost---------------------------------
close('all')
T=32
# ukazi4.m:99
N=256
# ukazi4.m:100
dT=T - 1
# ukazi4.m:101
TFD,f,t=specgram(sig,N,Fs,window(rectwin,T),dT,nargout=3)
# ukazi4.m:102
figure()
contour(t,f,abs(TFD))
axis('tight')
xlabel('t (s)','FontSize',12)
ylabel('f (Hz)','FontSize',12)
axis('tight')
title(cat('Spektrogram: T=',str(T),',N=',str(N),',dT=',str(dT)))
# asovno-frekvenna loljivost drugi: nija asovna loljivost, vija frekvenna loljivost--------------------------------
T=256
# ukazi4.m:110
N=256
# ukazi4.m:111
dT=T - 1
# ukazi4.m:112
TFD,f,t=specgram(sig,N,Fs,window(rectwin,T),dT,nargout=3)
# ukazi4.m:113
figure()
contour(t,f,abs(TFD))
axis('tight')
xlabel('t (s)','FontSize',12)
ylabel('f (Hz)','FontSize',12)
axis('tight')
title(cat('Spektrogram: T=',str(T),',N=',str(N),',dT=',str(dT)))
# Wigner-Villova asovno-frekvenna porazdelitev - skoraj idealna asovna in frekvenna loljivost----------------------------
TFD,t,f=tfrwv(sig,nargout=3)
# ukazi4.m:121
figure()
contour(t,f,abs(TFD),100)
axis('tight')
xlabel('t','FontSize',12)
ylabel('f','FontSize',12)
axis('tight')
title(cat('Wigner-Villova asovno-frekvenna porazdelitev'))
# Wigner-Villova asovno-frekvenna porazdelitev - SAMO REALNI DEL SIGNALA - Zrcaljenje preko frekce Fsamp/4 !!! ---------------------------
TFD,t,f=tfrwv(real(sig),nargout=3)
# ukazi4.m:129
figure()
contour(t,f,abs(TFD),100)
axis('tight')
xlabel('t','FontSize',12)
ylabel('f','FontSize',12)
axis('tight')
title(cat('Wigner-Villova asovno-frekvenna porazdelitev'))
# pseudo-Wigner-Villova asovno-frekvenna porazdelitev -----------------------------------------------------------------------
#    - okno v asovni domeni, ki prepreuje interference med asovno odmaknjenimi atomi
TFD,t,f=tfrpwv(sig,arange(1,Fs),Fs,hamming(31),nargout=3)
# ukazi4.m:139
figure()
contour(t,f,abs(TFD))
axis('tight')
xlabel('t','FontSize',12)
ylabel('f','FontSize',12)
axis('tight')
title(cat('Psevdo Wigner-Villova asovno-frekvenna porazdelitev'))
# pseudo-Wigner-Villova asovno-frekvenna porazdelitev - SAMO REALNI DEL SIGNALA - Zrcaljenje preko frekce Fsamp/4  !!! -----------------------------------------
#    - okno v asovni domeni, ki prepreuje interference med asovno odmaknjenimi atomi
TFD,t,f=tfrpwv(real(sig),arange(1,Fs),Fs,hamming(31),nargout=3)
# ukazi4.m:148
figure()
contour(t,f,abs(TFD))
axis('tight')
xlabel('t','FontSize',12)
ylabel('f','FontSize',12)
axis('tight')
title(cat('Psevdo Wigner-Villova asovno-frekvenna porazdelitev'))
'------------------------------------------------------------------------------------------------------------------------------'
# asovno-frekvenna loljivost in asovno-frekvenne porazdelitve - ve signalov skupaj

# # Primer 2: zgled z tirimi simetrino razporejenimi atomi v asovno-frekvenni ravnini.
Fs=1024
# ukazi4.m:161
sig=atoms(1*Fs,cat([1*Fs / 4,0.15,20,1],[3*Fs / 4,0.15,20,1],[1*Fs / 4,0.35,20,1],[3*Fs / 4,0.35,20,1]))
# ukazi4.m:162
figure()
plot(real(sig))

xlabel('t')
ylabel('amplituda')
axis('tight')
figure()
plot(abs(fft.fft(sig)))

xlabel('f')
ylabel('amplituda')
axis('tight')
#######################################################################
# asovno-frekvenna loljivost prvi: vija asovna loljivost, nija frekvenna loljivost---------------------------------
close('all')
T=32
# ukazi4.m:178
N=256
# ukazi4.m:179
dT=T - 1
# ukazi4.m:180
TFD,f,t=specgram(sig,N,Fs,window(rectwin,T),dT,nargout=3)
# ukazi4.m:181
figure()
imagesc(t,f[1:end() / 2],abs(TFD[1:end() / 2,:]))
axis('xy')
xlabel('t (s)','FontSize',12)
ylabel('f (Hz)','FontSize',12)
title(cat('Spektrogram: T=',str(T),',N=',str(N),',dT=',str(dT)))

T=256
# ukazi4.m:188
N=256
# ukazi4.m:189
dT=T - 1
# ukazi4.m:190
TFD,f,t=specgram(sig,N,Fs,window(rectwin,T),dT,nargout=3)
# ukazi4.m:191
figure()
imagesc(t,f[1:end() / 2],abs(TFD[1:end() / 2,:]))
axis('xy')
xlabel('t (s)','FontSize',12)
ylabel('f (Hz)','FontSize',12)
axis('tight')
title(cat('Spektrogram: T=',str(T),',N=',str(N),',dT=',str(dT)))
# Wigner-Villova asovno-frekvenna porazdelitev - skoraj idealna asovna in frekvenna loljivost----------------------------
TFD,t,f=tfrwv(sig,nargout=3)
# ukazi4.m:199
figure()
contour(t,f,abs(TFD))
axis('xy')
xlabel('t','FontSize',12)
ylabel('f','FontSize',12)
axis('tight')
title(cat('Wigner-Villova asovno-frekvenna porazdelitev'))
# pseudo-Wigner-Villova asovno-frekvenna porazdelitev -----------------------------------------------------------------------
#    - okno v asovni domeni, ki prepreuje interference med asovno odmaknjenimi atomi
TFD,t,f=tfrpwv(sig,arange(1,Fs),Fs,hamming(31),nargout=3)
# ukazi4.m:208
figure()
contour(t,f,abs(TFD))
axis('xy')
xlabel('t','FontSize',12)
ylabel('f','FontSize',12)
axis('tight')
title(cat('Psevdo Wigner-Villova asovno-frekvenna porazdelitev'))
# pseudo-Wigner-Villova asovno-frekvenna porazdelitev - SAMO REALNI DEL SIGNALA - Zrcaljenje preko frekce Fsamp/4 !!!! ----------------------------------------
#    - okno v asovni domeni, ki prepreuje interference med asovno odmaknjenimi atomi
TFD,t,f=tfrpwv(real(sig),arange(1,Fs),Fs,hamming(31),nargout=3)
# ukazi4.m:217
figure()
contour(t,f,abs(TFD))
axis('xy')
xlabel('t','FontSize',12)
ylabel('f','FontSize',12)
axis('tight')
title(cat('Psevdo Wigner-Villova asovno-frekvenna porazdelitev'))
# Choi-Williams asovno-frekvenna porazdelitev ------------------------------------------------------------------------------
#      - okno v asovni domeni, ki prepreuje interference med asovno odmaknjenimi atomi
#      - okno v frekvenni domeni, ki prepreuje interference med frekvenno odmaknjenimi atomi
TFD,t,f=tfrcw(sig,arange(1,Fs),Fs,hamming(31),hamming(31),nargout=3)
# ukazi4.m:228
figure()
imagesc(t,f,abs(TFD))
axis('xy')
xlabel('t','FontSize',12)
ylabel('f','FontSize',12)
axis('tight')
title(cat('Choi-Williams asovno-frekvenna porazdelitev'))
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
figure()
subplot(2,1,1)

plot(cat(arange(0,length(sig) - 1)) / Fs,real(sig))
axis('tight')
xlabel('t (s)')
ylabel('amplituda')
title('signal')

subplot(2,1,2)

plot(cat(arange(0,length(sig) - 1)) / Fs,dot(trenutnaFrekvenca,Fs))
xlabel('t (s)')
ylabel('f (Hz)')
title('trenutna frekvenca')
axis('tight')
# asovno-frekvenna loljivost prvi: vija asovna loljivost, nija frekvenna loljivost---------------------------------
close('all')
winT=32
# ukazi4.m:263
N=256
# ukazi4.m:264
dT=winT - 1
# ukazi4.m:265
TFD,f,t=specgram(sig,N,Fs,window(rectwin,winT),dT,nargout=3)
# ukazi4.m:266
figure()
contour(t,f[1:end() / 2],abs(TFD[1:end() / 2,:]))
axis('xy')
xlabel('t (s)','FontSize',12)
ylabel('f (Hz)','FontSize',12)
title(cat('Spektrogram: winT=',str(winT),',N=',str(N),',dT=',str(dT)))
# asovno-frekvenna loljivost drugi: nija asovna loljivost, vija frekvenna loljivost--------------------------------
winT=256
# ukazi4.m:273
N=256
# ukazi4.m:274
dT=winT - 1
# ukazi4.m:275
TFD,f,t=specgram(sig,N,Fs,window(rectwin,winT),dT,nargout=3)
# ukazi4.m:276
figure()
imagesc(t,f[1:end() / 2],abs(TFD[1:end() / 2,:]))
axis('xy')
xlabel('t (s)','FontSize',12)
ylabel('f (Hz)','FontSize',12)
axis('tight')
title(cat('Spektrogram: winT=',str(winT),',N=',str(N),',dT=',str(dT)))
# Wigner-Villova asovno-frekvenna porazdelitev - skoraj idealna asovna in frekvenna loljivost----------------------------
TFD,t,f=tfrwv(sig,arange(1,dot(T,Fs)),Fs,nargout=3)
# ukazi4.m:284
figure()
contour(t,f,abs(TFD))
axis('xy')
xlabel('t','FontSize',12)
ylabel('f','FontSize',12)
axis('tight')
title(cat('Wigner-Villova asovno-frekvenna porazdelitev'))
# pseudo-Wigner-Villova asovno-frekvenna porazdelitev -----------------------------------------------------------------------
#    - okno v asovni domeni, ki prepreuje interference med asovno odmaknjenimi atomi
TFD,t,f=tfrpwv(sig,arange(1,dot(T,Fs)),Fs,hamming(31),nargout=3)
# ukazi4.m:293
figure()
contour(t,f,abs(TFD))
axis('xy')
xlabel('t','FontSize',12)
ylabel('f','FontSize',12)
axis('tight')
title(cat('Psevdo Wigner-Villova asovno-frekvenna porazdelitev'))
# Choi-Williams asovno-frekvenna porazdelitev ------------------------------------------------------------------------------
#      - okno v asovni domeni, ki prepreuje interference med asovno odmaknjenimi atomi
#      - okno v frekvenni domeni, ki prepreuje interference med frekvenno odmaknjenimi atomi
TFD,t,f=tfrcw(sig,arange(1,dot(T,Fs)),Fs,hamming(25),hamming(25),nargout=3)
# ukazi4.m:303
figure()
contour(t,f,abs(TFD))
axis('xy')
xlabel('t','FontSize',12)
ylabel('f','FontSize',12)
axis('tight')
title(cat('Choi-Williams asovno-frekvenna porazdelitev'))
'------------------------------------------------------------------------------------------'
################################################################
# Primeri iz realnega ivljenja
clc
clear
close('all')
Fs=8000
# ukazi4.m:316

bits=8
# ukazi4.m:317

T=2
# ukazi4.m:318

#my_recorder=audiorecorder(Fs,bits,1)
# ukazi4.m:320
disp(cat('Snemam posnetek v dolini ',str(T),'s...'))
#recordblocking(my_recorder,T)
sig=sd.rec(T*Fs, Fs, 1)
# ukazi4.m:324
figure()
plot(cat(arange(1,dot(T,Fs))) / Fs,sig)
title('Posneti signal')
xlabel('t (s)')
ylabel('amplituda')
wavplay(sig,Fs,'sync')
sig=scipy.signal.hilbert(sig)
# ukazi4.m:328

# od tu dalje je pa dodajte kodo sami. Govorimo seveda o asovno-frekvenni analizi.
# POZOR! asovno-frekvenne porazdelitve so lahko raunsko precej precej porene.
# Priporoam majhne korake pri poveavi doline signala in smelo nastavljanje tevila frekvennih tok.