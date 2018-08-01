# 2-Linearni Sistemi in Konvolucija
import numpy as np
import scipy.signal
from matplotlib import cm  # color mapping
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.plotly as py
import plotly.graph_objs as go
from PIL import Image

'--------------------------------------------------------------------\n% LINEARNI SISTEMI IN KONVOLUCIJA\n% (C) ROSIS 2012, LSPO, FERI, Univerza v Mariboru, Slovenia \n% \n% Poglavje 1: 1D signali\n'
# Konvoluvcija: preprosti primeri z for zanko

# originalna formula za konvolucijo (samo vzroni del):

#             inf
#            ---
#     y(n) = \    x(k)*h(n-k)            za n = 0,1,2,...
#            /
#            ---
#             k=0

# v naem primeru je prvi element x oz. h na indeksu 1 (in ne na 0, kot je to v zgornji formuli), torej

#             inf
#            ---
#     y(n) = \    x(k+1)*h(n-k)          za n = 1,2,3,...
#            /
#            ---
#             k=0

# OPOMBA: pri n-k se vpliv postavitve zaetnega indeksa izniči: n+1 - (k+1) = n-k

# n je smiselno omejiti z zgornjo mejo len(x)+len(h)-1, saj so naprej same nile...
# zaradi h(n-k) mora biti n-k med 1 in len(h), torej mora biti k med n-len(h) in n-1, ampak samo za pozitivne n-k!
# zaradi x(k+1) mora teci k med 0 in len(x)-1

# ukazi.m:33
print("\n" * 80)  # clc
x = (1, 2, 3, 4, 3, 2, 1)
h = (1, 2, 1)

# ukazi.m:37
y = np.zeros(len(x) + len(h) - 1).tolist()  # dolžina izhoda, pretvorimo v python list za izris
for n in range(0, len(x) + len(h) - 1):  # V Pythonu je prvi element vektorja/seznama na indexu 0.
    print('...............')
    print('  n = {0}'.format(n))
    for k in range(max(n - len(h) + 1, 0), min(n + 1, len(x))):
        print('     n-k = {0}'.format(n - k))
        y[n] = y[n] + x[k] * h[n - k]

# ukazi.m:48
y2 = np.convolve(x, h).tolist()
# ukazi.m:50
plt.figure()
blue, = plt.plot(y, 'b-', linewidth=2, label='for zanka')
red, = plt.plot(y2, 'r:', linewidth=2, label='convolve')
plt.xlabel('vzorci')
plt.ylabel('amplituda')
plt.legend([blue, red], [blue.get_label(), red.get_label()])
'--------------------------------------------------------------------\n% Konvoluvcija: preprosti primeri s for zanko \n'
###########################################

print("\n" * 80)  # clc
plt.close('all')

# ukazi.m:65
x = [1] + np.zeros(25).tolist() + [2] + np.zeros(25).tolist() + [1] + np.zeros(25).tolist()
h = np.arange(0, 1.1, 0.1).tolist() + np.arange(1, -0.025, - 0.025).tolist()

# ukazi.m:68
f, ax = plt.subplots(2)
ax[0].plot(x, 'b', lineWidth=2)
ax[0].set_xlabel('vzorci')
ax[0].set_ylabel('amplituda')
ax[0].set_title('vhod (x)')
ax[1].plot(h, 'g', lineWidth=2)
ax[1].set_xlabel('vzorci')
ax[1].set_ylabel('amplituda')
ax[1].set_title('odziv (h)')
plt.tight_layout()

# ukazi.m:74
plt.figure()
y = np.zeros(len(x) + len(h) - 1).tolist()  # dolzina izhoda
for n in range(0, len(x) + len(h) - 1):
    for k in range(max(n - len(h) + 1, 0), min(n + 1, len(x))):
        y[n] = y[n] + x[k] * h[n - k]
    plt.cla()
    plt.plot(x, 'b', lineWidth=2)
    plt.plot((n - np.arange(len(h), 0, - 1)).tolist(), h[::-1], 'g', lineWidth=2)
    plt.plot(y, 'r', lineWidth=2)
    plt.xlabel('vzorci')
    plt.ylabel('amplituda')
    plt.title('vhod (x) - modro, odziv (h) - zeleno, izhod (y) - rdeče')
    plt.pause(0.01)
plt.close()

######################
# ukazi.m:91
f, ax = plt.subplots(3)
# subplot(3,1,1)
ax[0].plot(x, lineWidth=2)
ax[0].set_title('x')
ax[0].set_xlabel('vzorci')
ax[0].set_ylabel('amplituda')
# subplot(3,1,2)
ax[1].plot(h, 'g', lineWidth=2)
ax[1].set_title('h')
ax[1].set_xlabel('vzorci')
ax[1].set_ylabel('amplituda')
# subplot(3,1,3)
ax[2].plot(y, 'r', lineWidth=2)
ax[2].set_title('y')
ax[2].set_xlabel('vzorci')
ax[2].set_ylabel('amplituda')
plt.tight_layout()

###########################################
# ukazi.m:107
d = 1  # razmik impulzov v x
x = np.zeros(100)
x[0:100:d] = 1
h = np.arange(0, 1.1, 0.1).tolist() + np.arange(1, -0.025, -0.025).tolist()

# x = [ 0 0 0 0 0 1 0 0 0 0 0 0 2 0 0 0 0 0 3]
# h = [-1 1 -1]

plt.figure()
plt.subplot(2, 1, 1);
plt.plot(x, 'b', lineWidth=2)
plt.xlabel('vzorci');
plt.ylabel('amplituda');
plt.title('vhod (x)')
plt.subplot(2, 1, 2);
plt.plot(h, 'g', lineWidth=2)
plt.xlabel('vzorci');
plt.ylabel('amplituda');
plt.title('odziv (h)')
plt.tight_layout()

# ukazi.m:122
plt.figure()
y = np.zeros(len(x) + len(h) - 1).tolist()
for n in range(0, len(x) + len(h) - 1):
    for k in range(max(n - len(h) + 1, 0), min(n + 1, len(x))):
        y[n] = y[n] + x[k] * h[n - k]
    plt.cla()
    plt.plot(x, 'b', lineWidth=2)
    plt.plot((n - np.arange(len(h), 0, - 1)).tolist(), h[::-1], 'g', lineWidth=2)
    plt.plot(y, 'r', lineWidth=2)
    plt.xlabel('vzorci')
    plt.ylabel('amplituda')
    plt.title('vhod (x) - modro, odziv (h) - zeleno, izhod (y) - rdeče')
    plt.pause(0.01)

## #########
# ukazi.m:138
f, ax = plt.subplots(3)
# subplot(3,1,1)
ax[0].plot(x, lineWidth=2)
ax[0].set_title('x')
ax[0].set_xlabel('vzorci')
ax[0].set_ylabel('amplituda')
# subplot(3,1,2)
ax[1].plot(h, 'g', lineWidth=2)
ax[1].set_title('h')
ax[1].set_xlabel('vzorci')
ax[1].set_ylabel('amplituda')
# subplot(3,1,3)
ax[2].plot(y, 'r', lineWidth=2)
ax[2].set_title('y')
ax[2].set_xlabel('vzorci')
ax[2].set_ylabel('amplituda')
plt.tight_layout()

'--------------------------------------------------------------------' \
'% Konvoluvcija: preprosti primeri s funkcijo np.convolve()' \
'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'

# ukazi.m:156
x = np.zeros(50).tolist() + [1] + np.zeros(50).tolist()
h = np.arange(0, 1.1, 0.1).tolist() + np.arange(1, -0.025, -0.025).tolist()
# h = np.ones(22).tolist();

f, ax = plt.subplots(3)
# subplot(3,1,1)
ax[0].plot(x, lineWidth=2)
ax[0].set_title('x')
ax[0].set_xlabel('vzorci')
ax[0].set_ylabel('amplituda')
# subplot(3,1,2)
ax[1].plot(h, 'g', lineWidth=2)
ax[1].set_title('h')
ax[1].set_xlabel('vzorci')
ax[1].set_ylabel('amplituda')
# subplot(3,1,3)
ax[2].plot(np.convolve(x, h).tolist(), 'r', lineWidth=2)
ax[2].set_title('conv(x,h)')
ax[2].set_xlabel('vzorci')
ax[2].set_ylabel('amplituda')
plt.tight_layout()

###########################################
# ukazi.m:176
x = np.zeros(50).tolist() + [1] + np.zeros(25).tolist() + [1] + np.zeros(50).tolist()
h = np.arange(0, 1.1, 0.1).tolist() + np.arange(1, -0.025, -0.025).tolist()
# h = np.ones(22).tolist();

f, ax = plt.subplots(3)
# subplot(3,1,1)
ax[0].plot(x, lineWidth=2)
ax[0].set_title('x')
ax[0].set_xlabel('vzorci')
ax[0].set_ylabel('amplituda')
# subplot(3,1,2)
ax[1].plot(h, 'g', lineWidth=2)
ax[1].set_title('h')
ax[1].set_xlabel('vzorci')
ax[1].set_ylabel('amplituda')
# subplot(3,1,3)
ax[2].plot(np.convolve(x, h).tolist(), 'r', lineWidth=2)
ax[2].set_title('conv(x,h)')
ax[2].set_xlabel('vzorci')
ax[2].set_ylabel('amplituda')
plt.tight_layout()

###########################################
# Bolj kompleksen primer....
# ukazi.m:197
d = 20  # razmik impulzov v x
x = np.zeros(50).tolist() + [1] + np.zeros(d).tolist() + [1] + np.zeros(d).tolist() + [1] + np.zeros(50).tolist()
# h = np.arange(0,1.1,0.1).tolist() + np.arange(1,-0.025,-0.025).tolist()
h = np.random.random_sample(30).tolist()

f, ax = plt.subplots(3)
# subplot(3,1,1)
ax[0].plot(x, lineWidth=2)
ax[0].set_title('x')
ax[0].set_xlabel('vzorci')
ax[0].set_ylabel('amplituda')
# subplot(3,1,2)
ax[1].plot(h, 'g', lineWidth=2)
ax[1].set_title('h')
ax[1].set_xlabel('vzorci')
ax[1].set_ylabel('amplituda')
# subplot(3,1,3)
ax[2].plot(np.convolve(x, h).tolist(), 'r', lineWidth=2)
ax[2].set_title('conv(x,h)')
ax[2].set_xlabel('vzorci')
ax[2].set_ylabel('amplituda')
plt.tight_layout()

'--------------------------------------------------------------------'
# ALGEBRAI�NE LASTNOSTI KONVOLUCIJE
# KOMUTATIVNOST
#     f * g = g * f ,
# 
# ASOCIATIVNOST
#     f * (g * h) = (f * g) * h ,
# 
# DISTRIBUTIVNOST
#     f * (g + h) = (f * g) + (f * h) ,
# 
# ASOCIATIVNOST S SKALARNIM MNO�ENJEM
#     a (f * g) = (a f) * g = f * (a g) ,
# 
# KOMUTATIVNOST ###########################################
#     x * h = h * x ,

# ukazi.m:233
x = np.zeros(50).tolist() + [1] + np.zeros(50).tolist()
h = np.arange(0, 1.1, 0.1).tolist() + np.arange(1, -0.025, -0.025).tolist()

f, ax = plt.subplots(2)
ax[0].plot(np.convolve(x, h).tolist(), 'r', lineWidth=2)
ax[0].set_title('KOMUTATIVNOST KONVOLUCIJE: conv(x,h)')
ax[0].set_xlabel('vzorci')
ax[0].set_ylabel('amplituda')
ax[1].plot(np.convolve(h, x).tolist(), 'r', lineWidth=2)
ax[1].set_title('KOMUTATIVNOST KONVOLUCIJE: conv(h,x)')
ax[1].set_xlabel('vzorci')
ax[1].set_ylabel('amplituda')
plt.tight_layout()

# ASOCIATIVNOST ###########################################
#     g * (x * h) = (g * x) * h ,
# ukazi.m:247
x = np.zeros(50).tolist() + [1] + np.zeros(50).tolist()
h = np.arange(0, 1.1, 0.1).tolist() + np.arange(1, -0.025, -0.025).tolist()
g = np.sin(np.arange(0, np.pi, 0.1)).tolist()

f, ax = plt.subplots(2)
ax[0].plot(np.convolve(g, np.convolve(x, h)).tolist(), 'r', lineWidth=2)
ax[0].set_title('ASOCIATIVNOST KONVOLUCIJE: conv(g,conv(x,h))')
ax[0].set_xlabel('vzorci')
ax[0].set_ylabel('amplituda')
ax[1].plot(np.convolve(np.convolve(g, x), h), 'r', lineWidth=2)
ax[1].set_title('ASOCIATIVNOST KONVOLUCIJE: conv(conv(g,h),h)')
ax[1].set_xlabel('vzorci')
ax[1].set_ylabel('amplituda')
plt.tight_layout()

# DISTRIBUTIVNOST ###########################################
#     x * (g + h) = (x * g) + (x * h) ,
# ukazi.m:262
x = np.zeros(50).tolist() + [1] + np.zeros(50).tolist()
h = np.cos(np.arange(0, np.pi, 0.05)).tolist()
g = np.sin(np.arange(0, np.pi, 0.05)).tolist()

f, ax = plt.subplots(2)
ax[0].plot(np.convolve(x, (np.add(g, h))).tolist(), 'r', lineWidth=2)
ax[0].set_title('DISTRIBUTIVNOST KONVOLUCIJE:  conv(x,(g+h))')
ax[0].set_xlabel('vzorci')
ax[0].set_ylabel('amplituda')
ax[1].plot(np.add(np.convolve(x, g), np.convolve(x, h)).tolist(), 'r', lineWidth=2)
ax[1].set_title('DISTRIBUTIVNOST KONVOLUCIJE: conv(x,g) + conv(x,h)')
ax[1].set_xlabel('vzorci')
ax[1].set_ylabel('amplituda')
plt.tight_layout()

# ASOCIATIVNOST S SKALARNIM MNOŽENJEM ###########################################
#     a (x * h) = (a x) * h = x * (a h) ,
# ukazi.m:277
x = np.concatenate((np.zeros(50), [1], np.zeros(50)))  # Na tak način združimo polja numpyarray
h = np.sin(np.arange(0, np.pi, 0.05))
a = np.random.randn()

f, ax = plt.subplots(3)
# subplot(3,1,1)
ax[0].plot((a * np.convolve(x, h)).tolist(), 'r', lineWidth=2)
ax[0].set_title('ASOCIATIVNOST S SKALARNIM MNOŽENJEM: a*conv(x,h)')
ax[0].set_xlabel('vzorci')
ax[0].set_ylabel('amplituda')
# subplot(3,1,2)
ax[1].plot(np.convolve(a * x, h), 'g', lineWidth=2)
ax[1].set_title('ASOCIATIVNOST S SKALARNIM MNOŽENJEM: conv(a*x,h)')
ax[1].set_xlabel('vzorci')
ax[1].set_ylabel('amplituda')
# subplot(3,1,3)
ax[2].plot(np.convolve(x, a * h).tolist(), 'r', lineWidth=2)
ax[2].set_title('ASOCIATIVNOST S SKALARNIM MNOŽENJEM: conv(x,a*h)')
ax[2].set_xlabel('vzorci')
ax[2].set_ylabel('amplituda')
plt.tight_layout()

'--------------------------------------------------------------------' \
'% Konvoluvcija in govor' \
'% impulzni odzivi posameznih prostorov: http://www.voxengo.com/impulses/'
# '%%%%%% posnamemo govor %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import sounddevice as sd
# from scikits.audiolab import wavread Knjižnica še ni prevedena v Python 3
from scipy.io import wavfile
import os
import time

############################################################################################################################################################################ here
# ukazi.m:298
print("\n" * 80)  # clc
plt.close('all')
Fs = 44100
bits = 16
posnetek = sd.rec(5 * Fs, Fs, 2, blocking=True)
sd.play(posnetek, Fs)

# ukazi.m:308
f, ax = plt.subplots(2)
ax[0].plot(np.arange(0.0, len(posnetek)) / Fs, posnetek[:, 0], lineWidth=2)
ax[0].set_title('kanal 1')
ax[0].set_xlabel('čas (s)')
ax[0].set_ylabel('amplituda')
ax[1].plot(np.arange(0.0, len(posnetek)) / Fs, posnetek[:, 1], lineWidth=2)
ax[1].set_title('kanal 2')
ax[1].set_xlabel('čas (s)')
ax[1].set_ylabel('amplituda')
plt.tight_layout()

############# naložimo impulzni odziv sobe (dostopen na spletu) ########################
# impulzni odzivi posameznih prostorov: http://www.voxengo.com/impulses/
# ukazi.m:321
# h,Fs,bits=wavread('IMreverbs/Going Home.wav')
Fs, h = wavfile.read('../IMreverbs/Going Home.wav')
h = h / np.linalg.norm(h)
sd.play(50 * h, Fs)

# ukazi.m:325
f, ax = plt.subplots(2)
ax[0].plot(np.arange(0, len(h)) / Fs, h[:, 0], 'k', lineWidth=2)
ax[0].set_title('kanal 1')
ax[0].set_xlabel('čas (s)')
ax[0].set_ylabel('amplituda')
ax[1].plot(np.arange(0, len(h)) / Fs, h[:, 1], 'k', lineWidth=2)
ax[1].set_title('kanal 2')
ax[1].set_xlabel('čas (s)')
ax[1].set_ylabel('amplituda')
plt.tight_layout()

############# konvolucija v časovni domeni ########################
# ukazi.m:337
efekt = np.ndarray(shape=(len(posnetek) + len(h) - 1, 2))
tic = time.time()
# Proces traja ~30 sekund
efekt[:, 0] = np.convolve(posnetek[:, 0], h[:, 0])
efekt[:, 1] = np.convolve(posnetek[:, 1], h[:, 1])
toc = time.time() - tic
print('Pretekel čas: {0} sekund'.format(toc))
sd.play(efekt, Fs)

f, ax = plt.subplots(2)
ax[0].plot(np.arange(0, len(efekt)) / Fs, efekt[:, 0], 'r')
ax[0].plot(np.arange(0, len(posnetek)) / Fs, posnetek[:, 0])
ax[0].set_title('kanal 1')
ax[0].set_xlabel('čas (s)')
ax[0].set_ylabel('amplituda')
ax[1].plot(np.arange(0, len(efekt)) / Fs, efekt[:, 1], 'r')
ax[1].plot(np.arange(0, len(posnetek)) / Fs, posnetek[:, 1])
ax[1].set_title('kanal 2')
ax[1].set_xlabel('čas (s)')
ax[1].set_ylabel('amplituda')
plt.tight_layout()

############### konvolucija v frekvenčni domeni #######################
# ukazi.m:359
tic = time.time()
X = np.fft.fft(np.concatenate((posnetek[:, 0], np.zeros(len(h[:, 0]) - 1))))
Y = np.fft.fft(np.concatenate((h[:, 0], np.zeros(len(posnetek[:, 0]) - 1))))
efekt[:, 0] = np.fft.ifft(np.multiply(X, Y))
toc = time.time() - tic
print('Pretekel čas: {0} sekund'.format(toc))

tic = time.time()
X = np.fft.fft(np.concatenate((posnetek[:, 1], np.zeros(len(h[:, 1]) - 1))))
Y = np.fft.fft(np.concatenate((h[:, 1], np.zeros(len(posnetek[:, 1]) - 1))))
efekt[:, 1] = np.fft.ifft(np.multiply(X, Y))
toc = time.time() - tic
print('Pretekel čas: {0} sekund'.format(toc))

f, ax = plt.subplots(2)
ax[0].plot(np.arange(0, len(efekt)) / Fs, efekt[:, 0], 'r')
ax[0].plot(np.arange(0, len(posnetek)) / Fs, posnetek[:, 0])
ax[0].set_title('kanal 1')
ax[0].set_xlabel('čas (s)')
ax[0].set_ylabel('amplituda')
ax[1].plot(np.arange(0, len(efekt)) / Fs, efekt[:, 1], 'r')
ax[1].plot(np.arange(0, len(posnetek)) / Fs, posnetek[:, 1])
ax[1].set_title('kanal 2')
ax[1].set_xlabel('čas (s)')
ax[1].set_ylabel('amplituda')
plt.tight_layout()

sd.play(efekt, Fs)
################################################################################################################################################ here
'--------------------------------------------------------------------' \
'% kako pa je s konvolucijo v časovnem prostoru, če so signali dolgi'

h, Fs, bits = wavread('IMreverbs/Five columns.wav')
# ukazi.m:348
h = h / linalg.norm(h)
# ukazi.m:349
###### posnamemo govor ###########################################
# my_recorder=audiorecorder(Fs,bits,2)
# ukazi.m:352
# recordblocking(my_recorder,30)
# posnetek=getaudiodata(my_recorder)
posnetek = sd.rec(30 * Fs, Fs, 2)
# ukazi.m:354
sd.play(posnetek, Fs)
############### konvolucija v asovni domeni #######################
tic = time.time()
efekt[:, 0] = np.convolve(posnetek[:, 0], h[:, 0])
# ukazi.m:359
efekt[:, 1] = np.convolve(posnetek[:, 1], h[:, 1])
# ukazi.m:360
toc = time.time() - tic;
toc
sd.play(efekt, Fs)
############### konvolucija v frekvenni domeni #######################
tic = time.time()
X = np.fft.fft(([posnetek[:, 1]], [np.zeros(len(h[:, 1]) - 1, 1)]))
# ukazi.m:367
Y = np.fft.fft(([h[:, 1]], [np.zeros(len(posnetek[:, 1]) - 1, 1)]))
# ukazi.m:368
efekt[:, 1] = np.fft.ifft(multiply(X, Y))
# ukazi.m:369
toc = time.time() - tic;
toc
tic = time.time()
X = np.fft.fft(([posnetek[:, 2]], [np.zeros(len(h[:, 2]) - 1, 1)]))
# ukazi.m:373
Y = np.fft.fft(([h[:, 2]], [np.zeros(len(posnetek[:, 2]) - 1, 1)]))
# ukazi.m:374
efekt[:, 2] = np.fft.ifft(multiply(X, Y))
# ukazi.m:375
toc = time.time() - tic;
toc
sd.play(efekt, Fs)
'--------------------------------------------------------------------\n% Konvoluvcija in 3D zvok\n% impulzni odzivi posameznih pozicij v prostoru: http://recherche.ircam.fr/equipes/salles/listen/download.html \n%%%%%% posnamemo govor %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'

plt.close('all')
Fs = 44100
# ukazi.m:383
bits = 16
# ukazi.m:384
# my_recorder=audiorecorder(Fs,bits,2)
# ukazi.m:385
# recordblocking(my_recorder,5)
# posnetek=getaudiodata(my_recorder)
posnetek = sd.rec(5 * Fs, Fs, 2)  # dtype = 'int16'
# ukazi.m:387
sd.play(posnetek, Fs)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot((np.arange(1, len(posnetek))) / Fs, posnetek[:, 1])
plt.axis('tight')
plt.title('kanal 1')
plt.xlabel('as (s)')
plt.ylabel('amplituda')
plt.subplot(2, 1, 2)
plt.plot((np.arange(1, len(posnetek))) / Fs, posnetek[:, 2])
plt.axis('tight')
plt.title('kanal 2')
plt.xlabel('as (s)')
plt.ylabel('amplituda')
############# naloimo impulzni odziv 3D zvoka (dostopen na spletu) ########################
# impulzni odzivi posameznih prostorov: http://recherche.ircam.fr/equipes/salles/listen/download.html

load('.\\3Dsd.play\\IRC_1059_C_HRIR.mat')
elevation = 50
# ukazi.m:405

azimuth = 270
# ukazi.m:406

pos_err, pos_ind = min(abs(l_eq_hrir_S.elev_v - elevation) + abs(l_eq_hrir_S.azim_v - azimuth), nargout=2)
# ukazi.m:407
left_channel_IR = l_eq_hrir_S.content_m(pos_ind, np.arange())
# ukazi.m:409
right_channel_IR = r_eq_hrir_S.content_m(pos_ind, np.arange())
# ukazi.m:410
# sd.play(40*left_channel_IR,Fs);

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(np.dot((np.arange(1, len(left_channel_IR))) / Fs, 1000), left_channel_IR, 'k')
plt.axis('tight')
plt.title('levi kanal')
plt.xlabel('as (ms)')
plt.ylabel('amplituda')
plt.subplot(2, 1, 2)
plt.plot(np.dot((np.arange(1, len(right_channel_IR))) / Fs, 1000), right_channel_IR, 'k')
plt.axis('tight')
plt.title('desni kanal')
plt.xlabel('as (ms)')
plt.ylabel('amplituda')
############### konvolucija v frekvenni domeni #######################
clear('posnetek3D')
# convolveert stereo to mono sd.play
posnetekMONO = (posnetek[:, 1] + posnetek[:, 2]) / 2
# ukazi.m:428
# left channel
X = np.fft.fft(([posnetekMONO], [np.zeros(len(left_channel_IR) - 1, 1)]))
# ukazi.m:431
Y = np.fft.fft(([left_channel_IR.T], [np.zeros(len(posnetekMONO) - 1, 1)]))
# ukazi.m:432
posnetek3D[:, 1] = np.fft.ifft(multiply(X, Y))
# ukazi.m:433
# left channel
X = np.fft.fft(([posnetekMONO], [np.zeros(len(right_channel_IR) - 1, 1)]))
# ukazi.m:436
Y = np.fft.fft(([right_channel_IR.T], [np.zeros(len(posnetekMONO) - 1, 1)]))
# ukazi.m:437
posnetek3D[:, 2] = np.fft.ifft(multiply(X, Y))
# ukazi.m:438
plt.figure()
plt.subplot(2, 1, 1)
plt.plot((np.arange(1, len(posnetek3D))) / Fs, posnetek3D[:, 1], 'r')

plt.plot((np.arange(1, len(posnetek))) / Fs, posnetek[:, 1])
plt.axis('tight')
plt.title('kanal 1')
plt.xlabel('as (s)')
plt.ylabel('amplituda')
plt.subplot(2, 1, 2)
plt.plot((np.arange(1, len(posnetek3D))) / Fs, posnetek3D[:, 2], 'r')

plt.plot((np.arange(1, len(posnetek))) / Fs, posnetek[:, 2])
plt.axis('tight')
plt.title('kanal 2')
plt.xlabel('as (s)')
plt.ylabel('amplituda')
sd.play(posnetek3D, Fs)
'--------------------------------------------------------------------\n% AUDIO EFEKT DISTORTION\n% (C) ROSIS, LSPO, FERI, Univerza v Mariboru, Slovenia \n--------------------------------------------------------------------\n'
## ################## akustina kitara
y, Fs, nbits = wavread('SotW.wav', nargout=3)
# ukazi.m:456
player = audioplayer(y, Fs)
# ukazi.m:457
play(player)
stop(player)
'--------------------------------------------------------------------\n%% %%%%%%%%%%%%%%%%%% efekt distortion %%%%%%%%%%%%%%%%%%\n%            y = vhodni signal\n%            Fs = Vzorevalna frekvenca\n'
A = 20
# ukazi.m:464
clear('yf')
dy = distortion(A, y)
# ukazi.m:466
dplayer = audioplayer(dy, Fs)
# ukazi.m:468
play(dplayer)
stop(player)
'--------------------------------------------------------------------\n% LINEARNI SISTEMI IN KONVOLUCIJA\n% (C) ROSIS, LSPO, FERI, Univerza v Mariboru, Slovenia \n% \n% Poglavje 2: Slike (2D signali)\n--------------------------------------------------------------------\n% SOBELOV OPERATOR\n'
plt.close('all')
A = imread('Valve_original.png')
# ukazi.m:476
plt.figure()
image(A)
plt.title('originalna slika')
Sobel_x = np.dot((1, 2, 1).T, (+ 1, 0, - 1))
# ukazi.m:480
Sobel_y = np.dot((+ 1, 0, - 1).T, (1, 2, 1))
# ukazi.m:481
B_x[:, :, 1] = convolve2(double(A[:, :, 1]), Sobel_x)
# ukazi.m:483
B_x[:, :, 2] = convolve2(double(A[:, :, 2]), Sobel_x)
# ukazi.m:484
B_x[:, :, 3] = convolve2(double(A[:, :, 3]), Sobel_x)
# ukazi.m:485+
B_x = uint8(B_x)
# ukazi.m:486
# B_x = uint8( (B_x - min(B_x(:))) / (max(B_x(:)) - min(B_x(:))) *255 );
plt.figure()
image(B_x)
plt.title('Sobelov operator v x smeri')
B_y[:, :, 1] = convolve2(double(A[:, :, 1]), Sobel_y)
# ukazi.m:491
B_y[:, :, 2] = convolve2(double(A[:, :, 2]), Sobel_y)
# ukazi.m:492
B_y[:, :, 3] = convolve2(double(A[:, :, 3]), Sobel_y)
# ukazi.m:493
B_y = uint8(B_y)
# ukazi.m:494
# B_y = uint8( (B_y - min(B_y(:))) / (max(B_y(:)) - min(B_y(:))) *255 );
plt.figure()
image(B_y)
plt.title('Sobelov operator v y smeri')
'--------------------------------------------------------------------\n% SOBELOV OPERATOR - drugi\n'
plt.close('all')
C = imread('Bikesgray.jpg')
# ukazi.m:502
plt.figure()
imagesc(C, (0, 255))
colormap(gray)
plt.title('Originalna slika')
Sobel_x = np.dot((1, 2, 1).T, (+ 1, 0, - 1))
# ukazi.m:506
Sobel_y = np.dot((+ 1, 0, - 1).T, (1, 2, 1))
# ukazi.m:507
D_x = convolve2(double(C), Sobel_x)
# ukazi.m:509
# D_x = uint8(D_x);
plt.figure()
imagesc(D_x)
colormap(gray)
plt.title('Sobelov operator v x smeri')
D_y = convolve2(double(C), Sobel_y)
# ukazi.m:515
# D_y = uint8(D_y);
plt.figure()
imagesc(D_y)
colormap(gray)
plt.title('Sobelov operator v y smeri')
plt.figure()
imagesc(sqrt(double(D_x ** 2 + D_y ** 2)))
colormap(gray)
plt.title('zdruen Sobelov operator v x in y smeri')
W_xy = sqrt(double(D_x ** 2 + D_y ** 2))
# ukazi.m:523
W_xy = uint8(abs(255 - W_xy))
# ukazi.m:524
W_xy[W_xy > 140] = 255
# ukazi.m:525
plt.figure()
imagesc(W_xy)
colormap(gray)
plt.title('zdruen Sobelov operator v x in y smeri')
'--------------------------------------------------------------------\n% Laplacov operator\n'
plt.close('all')
C = imread('Bikesgray.jpg')
# ukazi.m:531
plt.figure()
imagesc(C, (0, 255))
colormap(gray)
Laplace = np.ndarray(([0, - 1, 0], [- 1, 4, - 1], [0, - 1, 0]))
# ukazi.m:534
D_x = convolve2(double(C), Laplace)
# ukazi.m:536
plt.figure()
imagesc(D_x)
colormap(gray)
'--------------------------------------------------------------------\n% Glajenje slike - Gaussov operator\n'
plt.close('all')
C = imread('Bikesgray.jpg')
# ukazi.m:541
plt.figure()
imagesc(C, (0, 255))
colormap(gray)
plt.title('originalna slika')
# sliko zgladimo z Gausovim operatorjem velikosti 5x5
Gauss = np.dot(1 / 159, ([2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]))
# ukazi.m:546
plt.figure()
surf(Gauss)
plt.axis('tight')
plt.title('2D Gaussova funkcija')
D_x = convolve2(double(C), Gauss)
# ukazi.m:550
plt.figure()
imagesc(D_x)
colormap(gray)
plt.title('zglajena slika: Gauss 5x5')
####### Poljubna velikost Gaussovega operatorja NxN
N = 25
# ukazi.m:555
w = gausswin(N)
# ukazi.m:556
Gauss = np.dot(w, w.T)
# ukazi.m:557
Gauss = Gauss / sum(ravel(w))
# ukazi.m:557
plt.figure()
surf(Gauss)
plt.axis('tight')
plt.title('2D Gaussova funkcija')
D_x = convolve2(double(C), Gauss)
# ukazi.m:561
plt.figure()
imagesc(D_x)
colormap(gray)
plt.title(('zglajena slika: Gauss ', str(N), 'x', str(N)))