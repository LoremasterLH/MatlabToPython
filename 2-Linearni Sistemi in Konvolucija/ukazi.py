# Author: Martin Konečnik
# Contact: martin.konecnik@gmail.com
# Licenced under MIT

# 2-Linearni Sistemi in Konvolucija
import numpy as np
from scipy.signal import convolve2d as conv2
from scipy.signal.windows import gaussian as gausswin
import pylab as pylab
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Potrebni za del z zvokom
import sounddevice as sd
from scipy.io import wavfile
import time

# --------------------------------------------------------------------
# LINEARNI SISTEMI IN KONVOLUCIJA
# (C) ROSIS 2012, LSPO, FERI, Univerza v Mariboru, Slovenia
#
# Poglavje 1: 1D signali
# --------------------------------------------------------------------
# Konvolucija: preprosti primeri z for zanko
#
# originalna formula za konvolucijo (samo vzročni del):
#
#             inf
#            ---
#     y(n) = \    x(k)*h(n-k)            za n = 0,1,2,...
#            /
#            ---
#             k=0
#
# v našem primeru je prvi element x oz. h na indeksu 1 (in ne na 0, kot je to v zgornji formuli), torej
#
#             inf
#            ---
#     y(n) = \    x(k+1)*h(n-k)          za n = 1,2,3,...
#            /
#            ---
#             k=0
#
# OPOMBA: pri n-k se vpliv postavitve začetnega indeksa izniči: n+1 - (k+1) = n-k
#
# n je smiselno omejiti z zgornjo mejo len(x)+len(h)-1, saj so naprej same ničle...
# zaradi h(n-k) mora biti n-k med 1 in len(h), torej mora biti k med n-len(h) in n-1, ampak samo za pozitivne n-k!
# zaradi x(k+1) mora teči k med 0 in len(x)-1

# ukazi.m:33 -- Note: Preverjeno z Matlab
print("\n" * 80)  # clc - Python nima funkcionalnosti, ki bi konsistentno počistila konzolo. Lahko izbrišeš.

x = (1, 2, 3, 4, 3, 2, 1)
h = (1, 2, 1)

y = np.zeros(len(x) + len(h) - 1).tolist()  # dolžina izhoda, pretvorimo v python list za izris
for n in range(0, len(x) + len(h) - 1):  # V Pythonu je prvi element na indexu 0.
    print('...............')
    print('  n = {0}'.format(n))
    for k in range(max(n - len(h) + 1, 0), min(n + 1, len(x))):
        print('     n-k = {0}'.format(n - k))
        y[n] = y[n] + x[k] * h[n - k]

y2 = np.convolve(x, h).tolist()
plt.figure()
blue, = plt.plot(y, 'b-', linewidth=2, label='for zanka')
red, = plt.plot(y2, 'r:', linewidth=2, label='conv')
plt.xlabel('vzorci')
plt.ylabel('amplituda')
plt.legend([blue, red], [blue.get_label(), red.get_label()])

# ukazi.m:59
# --------------------------------------------------------------------
# Konvoluvcija: preprosti primeri s for zanko
# ##########################################

# ukazi.m:63 -- Note: Preverjeno z Matlab
print("\n" * 80)  # clc
plt.close('all')
x = [1] + np.zeros(25).tolist() + [2] + np.zeros(25).tolist() + [1] + np.zeros(25).tolist()
h = np.arange(0, 1.1, 0.1).tolist() + np.arange(1, -0.025, - 0.025).tolist()

# ukazi.m:68
fig, ax = plt.subplots(2)
ax[0].plot(x, 'b', lineWidth=2)
ax[0].set_xlabel('vzorci')
ax[0].set_ylabel('amplituda')
ax[0].set_title('Vhod (x)')

ax[1].plot(h, 'g', lineWidth=2)
ax[1].set_xlabel('vzorci')
ax[1].set_ylabel('amplituda')
ax[1].set_title('Odziv (h)')
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
    plt.title('Vhod (x) - modro, odziv (h) - zeleno, izhod (y) - rdeče')
    plt.pause(0.01)
plt.close()

# ukazi.m:90 -- Note: Preverjeno z Matlab
######################
fig, ax = plt.subplots(3)

ax[0].plot(x, lineWidth=2)
ax[0].set_xlabel('vzorci')
ax[0].set_ylabel('amplituda')
ax[0].set_title('x')

ax[1].plot(h, 'g', lineWidth=2)
ax[1].set_xlabel('vzorci')
ax[1].set_ylabel('amplituda')
ax[1].set_title('h')

ax[2].plot(y, 'r', lineWidth=2)
ax[2].set_xlabel('vzorci')
ax[2].set_ylabel('amplituda')
ax[2].set_title('y')
plt.tight_layout()

# ukazi.m:106 -- Note: Preverjeno z Matlab
###########################################
d = 1  # razmik impulzov v x
x = np.zeros(100)
x[0:100:d] = 1
h = np.arange(0, 1.1, 0.1).tolist() + np.arange(1, -0.025, -0.025).tolist()

# x = [ 0 0 0 0 0 1 0 0 0 0 0 0 2 0 0 0 0 0 3]
# h = [-1 1 -1]

fig, ax = plt.subplots(2)

ax[0].plot(x, 'b', lineWidth=2)
ax[0].set_xlabel('vzorci')
ax[0].set_ylabel('amplituda')
ax[0].set_title('Vhod (x)')

ax[1].plot(h, 'g', lineWidth=2)
ax[1].set_xlabel('vzorci')
ax[1].set_ylabel('amplituda')
ax[1].set_title('Odziv (h)')
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
# ukazi.m:138 -- Note: Preverjeno z Matlab
fig, ax = plt.subplots(3)

ax[0].plot(x, lineWidth=2)
ax[0].set_title('x')
ax[0].set_xlabel('vzorci')
ax[0].set_ylabel('amplituda')

ax[1].plot(h, 'g', lineWidth=2)
ax[1].set_title('h')
ax[1].set_xlabel('vzorci')
ax[1].set_ylabel('amplituda')

ax[2].plot(y, 'r', lineWidth=2)
ax[2].set_title('y')
ax[2].set_xlabel('vzorci')
ax[2].set_ylabel('amplituda')
plt.tight_layout()

# ukazi.m:152
# --------------------------------------------------------------------
# Konvoluvcija: preprosti primeri s funkcijo np.convolve()
# ###########################################

# ukazi.m:156 -- Note: Preverjeno z Matlab
x = np.zeros(50).tolist() + [1] + np.zeros(50).tolist()
h = np.arange(0, 1.1, 0.1).tolist() + np.arange(1, -0.025, -0.025).tolist()
# h = np.ones(22).tolist();

fig, ax = plt.subplots(3)

ax[0].plot(x, lineWidth=2)
ax[0].set_title('x')
ax[0].set_xlabel('vzorci')
ax[0].set_ylabel('amplituda')

ax[1].plot(h, 'g', lineWidth=2)
ax[1].set_title('h')
ax[1].set_xlabel('vzorci')
ax[1].set_ylabel('amplituda')

ax[2].plot(np.convolve(x, h).tolist(), 'r', lineWidth=2)
ax[2].set_title('conv(x,h)')
ax[2].set_xlabel('vzorci')
ax[2].set_ylabel('amplituda')
plt.tight_layout()

# ukazi.m:175 -- Note: Preverjeno z Matlab
###########################################
x = np.zeros(50).tolist() + [1] + np.zeros(25).tolist() + [1] + np.zeros(50).tolist()
h = np.arange(0, 1.1, 0.1).tolist() + np.arange(1, -0.025, -0.025).tolist()
# h = np.ones(22).tolist();

fig, ax = plt.subplots(3)

ax[0].plot(x, lineWidth=2)
ax[0].set_title('x')
ax[0].set_xlabel('vzorci')
ax[0].set_ylabel('amplituda')

ax[1].plot(h, 'g', lineWidth=2)
ax[1].set_title('h')
ax[1].set_xlabel('vzorci')
ax[1].set_ylabel('amplituda')

ax[2].plot(np.convolve(x, h).tolist(), 'r', lineWidth=2)
ax[2].set_title('conv(x,h)')
ax[2].set_xlabel('vzorci')
ax[2].set_ylabel('amplituda')
plt.tight_layout()

# ukazi.m:195 -- Note: Preverjeno z Matlab
###########################################
# Bolj kompleksen primer....
d = 20  # razmik impulzov v x
x = np.zeros(50).tolist() + [1] + np.zeros(d).tolist() + [1] + np.zeros(d).tolist() + [1] + np.zeros(50).tolist()
# h = np.arange(0,1.1,0.1).tolist() + np.arange(1,-0.025,-0.025).tolist()
h = np.random.random_sample(30).tolist()

fig, ax = plt.subplots(3)
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

# ukazi.m:216
# --------------------------------------------------------------------
# ALGEBRAIČNE LASTNOSTI KONVOLUCIJE
# KOMUTATIVNOST
#     f * g = g * f ,
# 
# ASOCIATIVNOST
#     f * (g * h) = (f * g) * h ,
# 
# DISTRIBUTIVNOST
#     f * (g + h) = (f * g) + (f * h) ,
# 
# ASOCIATIVNOST S SKALARNIM MNOŽENJEM
#     a (f * g) = (a f) * g = f * (a g) ,
# 
# KOMUTATIVNOST ###########################################
#     x * h = h * x ,

# ukazi.m:233 -- Note: Preverjeno z Matlab
x = np.zeros(50).tolist() + [1] + np.zeros(50).tolist()
h = np.arange(0, 1.1, 0.1).tolist() + np.arange(1, -0.025, -0.025).tolist()

fig, ax = plt.subplots(2)
ax[0].plot(np.convolve(x, h).tolist(), 'r', lineWidth=2)
ax[0].set_title('KOMUTATIVNOST KONVOLUCIJE: conv(x,h)')
ax[0].set_xlabel('vzorci')
ax[0].set_ylabel('amplituda')
ax[1].plot(np.convolve(h, x).tolist(), 'r', lineWidth=2)
ax[1].set_title('KOMUTATIVNOST KONVOLUCIJE: conv(h,x)')
ax[1].set_xlabel('vzorci')
ax[1].set_ylabel('amplituda')
plt.tight_layout()

# ukazi.m:247 -- Note: Preverjeno z Matlab
# ASOCIATIVNOST ###########################################
#     g * (x * h) = (g * x) * h ,
x = np.zeros(50).tolist() + [1] + np.zeros(50).tolist()
h = np.arange(0, 1.1, 0.1).tolist() + np.arange(1, -0.025, -0.025).tolist()
g = np.sin(np.arange(0, np.pi, 0.1)).tolist()

fig, ax = plt.subplots(2)
ax[0].plot(np.convolve(g, np.convolve(x, h)).tolist(), 'r', lineWidth=2)
ax[0].set_title('ASOCIATIVNOST KONVOLUCIJE: conv(g,conv(x,h))')
ax[0].set_xlabel('vzorci')
ax[0].set_ylabel('amplituda')
ax[1].plot(np.convolve(np.convolve(g, x), h), 'r', lineWidth=2)
ax[1].set_title('ASOCIATIVNOST KONVOLUCIJE: conv(conv(g,h),h)')
ax[1].set_xlabel('vzorci')
ax[1].set_ylabel('amplituda')
plt.tight_layout()

# ukazi.m:260 -- Note: Preverjeno z Matlab
# DISTRIBUTIVNOST ###########################################
#     x * (g + h) = (x * g) + (x * h) ,
x = np.zeros(50).tolist() + [1] + np.zeros(50).tolist()
h = np.cos(np.arange(0, np.pi, 0.05)).tolist()
g = np.sin(np.arange(0, np.pi, 0.05)).tolist()

fig, ax = plt.subplots(2)
ax[0].plot(np.convolve(x, (np.add(g, h))).tolist(), 'r', lineWidth=2)
ax[0].set_title('DISTRIBUTIVNOST KONVOLUCIJE:  conv(x,(g+h))')
ax[0].set_xlabel('vzorci')
ax[0].set_ylabel('amplituda')
ax[1].plot(np.add(np.convolve(x, g), np.convolve(x, h)).tolist(), 'r', lineWidth=2)
ax[1].set_title('DISTRIBUTIVNOST KONVOLUCIJE: conv(x,g) + conv(x,h)')
ax[1].set_xlabel('vzorci')
ax[1].set_ylabel('amplituda')
plt.tight_layout()

# ukazi.m:275 -- Note: Preverjeno z Matlab
# ASOCIATIVNOST S SKALARNIM MNOŽENJEM ###########################################
#     a (x * h) = (a x) * h = x * (a h) ,
x = np.concatenate((np.zeros(50), [1], np.zeros(50)))  # Na tak način združimo polja numpyarray
h = np.sin(np.arange(0, np.pi, 0.05))
a = np.random.randn()

fig, ax = plt.subplots(3)

ax[0].plot((a * np.convolve(x, h)).tolist(), 'r', lineWidth=2)
ax[0].set_title('ASOCIATIVNOST S SKALARNIM MNOŽENJEM: a*conv(x,h)')
ax[0].set_xlabel('vzorci')
ax[0].set_ylabel('amplituda')

ax[1].plot(np.convolve(a * x, h), 'r', lineWidth=2)
ax[1].set_title('ASOCIATIVNOST S SKALARNIM MNOŽENJEM: conv(a*x,h)')
ax[1].set_xlabel('vzorci')
ax[1].set_ylabel('amplituda')

ax[2].plot(np.convolve(x, a * h).tolist(), 'r', lineWidth=2)
ax[2].set_title('ASOCIATIVNOST S SKALARNIM MNOŽENJEM: conv(x,a*h)')
ax[2].set_xlabel('vzorci')
ax[2].set_ylabel('amplituda')
plt.tight_layout()

# ukazi.m:294
# --------------------------------------------------------------------
# Konvoluvcija in govor
# impulzni odzivi posameznih prostorov: http://www.voxengo.com/impulses
# ###### posnamemo govor ###########################################

# ukazi.m:298 -- Note: Preverjeno z Matlab
print("\n" * 80)  # clc
plt.close('all')
Fs = 44100
bits = 16
posnetek = sd.rec(5 * Fs, Fs, 2, blocking=True)
sd.play(posnetek, Fs)

# ukazi.m:308
fig, ax = plt.subplots(2)
ax[0].plot(np.arange(0.0, len(posnetek)) / Fs, posnetek[:, 0], lineWidth=0.4)
ax[0].set_title('kanal 1')
ax[0].set_xlabel('čas (s)')
ax[0].set_ylabel('amplituda')
ax[1].plot(np.arange(0.0, len(posnetek)) / Fs, posnetek[:, 1], lineWidth=0.4)
ax[1].set_title('kanal 2')
ax[1].set_xlabel('čas (s)')
ax[1].set_ylabel('amplituda')
plt.tight_layout()

# ukazi.m:318 -- Note: Preverjeno z Matlab
# ############ naložimo impulzni odziv sobe (dostopen na spletu) ########################
# impulzni odzivi posameznih prostorov: http://www.voxengo.com/impulses/
# h,Fs,bits=wavread('IMreverbs/Going Home.wav')
# Če datoteke ne najde, verjetno išče v mapi prej odprte. Najlažje se reši, če zapreš IDE in ponovno odpreš to datoteko,
# ali spremeniš pot.
Fs, h = wavfile.read('./2-Linearni Sistemi in Konvolucija/IMreverbs/Going Home.wav')
h = h / np.linalg.norm(h)
sd.play(50 * h, Fs)

# ukazi.m:325
fig, ax = plt.subplots(2)
ax[0].plot(np.arange(0, len(h)) / Fs, h[:, 0], 'k', lineWidth=0.4)
ax[0].set_title('kanal 1')
ax[0].set_xlabel('čas (s)')
ax[0].set_ylabel('amplituda')
ax[1].plot(np.arange(0, len(h)) / Fs, h[:, 1], 'k', lineWidth=0.4)
ax[1].set_title('kanal 2')
ax[1].set_xlabel('čas (s)')
ax[1].set_ylabel('amplituda')
plt.tight_layout()

# ukazi.m:336 -- Note: Preverjeno z Matlab
# ############ konvolucija v časovni domeni ########################
efekt = np.ndarray(shape=(len(posnetek) + len(h) - 1, 2))
tic = time.time()
# Proces je počasen, tako da ne skrbet, da traja predolgo. Cca 2 min pri meni.
efekt[:, 0] = np.convolve(posnetek[:, 0], h[:, 0])
efekt[:, 1] = np.convolve(posnetek[:, 1], h[:, 1])
toc = time.time() - tic
print('Pretekel čas: {0} sekund'.format(toc))
sd.play(efekt, Fs)

fig, ax = plt.subplots(2)
ax[0].plot(np.arange(0, len(efekt)) / Fs, efekt[:, 0], 'r', lineWidth=0.4)
ax[0].plot(np.arange(0, len(posnetek)) / Fs, posnetek[:, 0], lineWidth=0.4)
ax[0].set_title('kanal 1')
ax[0].set_xlabel('čas (s)')
ax[0].set_ylabel('amplituda')
ax[1].plot(np.arange(0, len(efekt)) / Fs, efekt[:, 1], 'r', lineWidth=0.4)
ax[1].plot(np.arange(0, len(posnetek)) / Fs, posnetek[:, 1], lineWidth=0.4)
ax[1].set_title('kanal 2')
ax[1].set_xlabel('čas (s)')
ax[1].set_ylabel('amplituda')
plt.tight_layout()

# ukazi.m:356 -- Note: Preverjeno z Matlab
# ############## konvolucija v frekvenčni domeni #######################
tic = time.time()
X = np.fft.fft(np.concatenate((posnetek[:, 0], np.zeros(len(h[:, 0]) - 1))))
Y = np.fft.fft(np.concatenate((h[:, 0], np.zeros(len(posnetek[:, 0]) - 1))))
efekt = np.empty([X.size, 2])  # Initialise new np array as python does not do this automatically to fit data.
efekt[:, 0] = np.fft.ifft(np.multiply(X, Y))
toc = time.time() - tic
print('Pretekel čas: {0} sekund'.format(toc))

tic = time.time()
X = np.fft.fft(np.concatenate((posnetek[:, 1], np.zeros(len(h[:, 1]) - 1))))
Y = np.fft.fft(np.concatenate((h[:, 1], np.zeros(len(posnetek[:, 1]) - 1))))
efekt[:, 1] = np.fft.ifft(np.multiply(X, Y))
toc = time.time() - tic
print('Pretekel čas: {0} sekund'.format(toc))

fig, ax = plt.subplots(2)
ax[0].plot(np.arange(0, len(efekt)) / Fs, efekt[:, 0], 'r', lineWidth=0.4)
ax[0].plot(np.arange(0, len(posnetek)) / Fs, posnetek[:, 0], lineWidth=0.4)
ax[0].set_title('kanal 1')
ax[0].set_xlabel('čas (s)')
ax[0].set_ylabel('amplituda')
ax[1].plot(np.arange(0, len(efekt)) / Fs, efekt[:, 1], 'r', lineWidth=0.4)
ax[1].plot(np.arange(0, len(posnetek)) / Fs, posnetek[:, 1], lineWidth=0.4)
ax[1].set_title('kanal 2')
ax[1].set_xlabel('čas (s)')
ax[1].set_ylabel('amplituda')
plt.tight_layout()

sd.play(efekt, Fs)

# ukazi.m:385
# --------------------------------------------------------------------
# kako pa je s konvolucijo v časovnem prostoru, če so signali dolgi
Fs, h = wavfile.read('./2-Linearni Sistemi in Konvolucija/IMreverbs/Five columns.wav')
h = h / np.linalg.norm(h)

# ukazi.m:391 -- Note: Preverjeno z Matlab
# ##### posnamemo govor ###########################################
# 30 sekund
posnetek = sd.rec(30 * Fs, Fs, 2, blocking=True)
sd.play(posnetek, Fs)

# ukazi.m:397 -- Note: Preverjeno z Matlab
# ############## konvolucija v časovni domeni #######################
tic = time.time()
# Lahko traja nekaj minut. Cca 5 min pri meni.
efekt = np.empty([X.size, 2])  # Initialise new np array as python does not do this automatically to fit data.
efekt[:, 0] = np.convolve(posnetek[:, 0], h[:, 0])
efekt[:, 1] = np.convolve(posnetek[:, 1], h[:, 1])
toc = time.time() - tic
print('Pretekel čas: {0} sekund'.format(toc))

sd.play(efekt, Fs)

# ukazi.m:405 -- Note: Rezultat je pričakovan, vendar ni preverjeno, ker v Matlabu ni delalo.
# ############## konvolucija v frekvenni domeni #######################
tic = time.time()
X = np.fft.fft(np.concatenate((posnetek[:, 0], np.zeros(len(h[:, 0]) - 1))))
Y = np.fft.fft(np.concatenate((h[:, 0], np.zeros(len(posnetek[:, 0]) - 1))))
efekt = np.empty([X.size, 2])  # Initialise new np array as python does not do this automatically to fit data.
efekt[:, 0] = np.fft.ifft(np.multiply(X, Y))
toc = time.time() - tic
print('Pretekel čas: {0} sekund'.format(toc))
tic = time.time()
X = np.fft.fft(np.concatenate((posnetek[:, 1], np.zeros(len(h[:, 1]) - 1))))
Y = np.fft.fft(np.concatenate((h[:, 1], np.zeros(len(posnetek[:, 1]) - 1))))
efekt[:, 1] = np.fft.ifft(np.multiply(X, Y))
toc = time.time() - tic
print('Pretekel čas: {0} sekund'.format(toc))

sd.play(efekt, Fs)

# ukazi.m:420
# --------------------------------------------------------------------
# Konvoluvcija in 3D zvok
# impulzni odzivi posameznih pozicij v prostoru: http://recherche.ircam.fr/equipes/salles/listen/download.html
# ###### posnamemo govor ###########################################

# ukazi.m:425 -- Note: Preverjeno z Matlab
print("\n" * 80)  # clc
plt.close('all')
Fs = 44100
posnetek = sd.rec(5 * Fs, Fs, 2, blocking=True)  # Za spremembo bitne ločljivosti nastavimo dtype (privzeto 16 bitna)
sd.play(posnetek, Fs)

# ukazi.m:434
fig, ax = plt.subplots(2)
ax[0].plot(np.arange(0, len(posnetek[:, 0])) / Fs, posnetek[:, 0], LineWidth=0.4)
ax[0].set_title('kanal 1')
ax[0].set_xlabel('čas (s)')
ax[0].set_ylabel('amplituda')

ax[1].plot(np.arange(0, len(posnetek[:, 1])) / Fs, posnetek[:, 1], LineWidth=0.4)
ax[1].set_title('kanal 2')
ax[1].set_xlabel('čas (s)')
ax[1].set_ylabel('amplituda')
plt.tight_layout()

# ukazi.m:444 -- Note: Not yet implemented due to having to read an unfamiliar file-type, continues at line 594
# ############ naložimo impulzni odziv 3D zvoka (dostopen na spletu) ########################
# impulzni odzivi posameznih prostorov: http://recherche.ircam.fr/equipes/salles/listen/download.html

load('.\\3Dsd.play\\IRC_1059_C_HRIR.mat')
elevation = 50
azimuth = 270
pos_err, pos_ind = min(abs(l_eq_hrir_S.elev_v - elevation) + abs(l_eq_hrir_S.azim_v - azimuth), nargout=2)
left_channel_IR = l_eq_hrir_S.content_m(pos_ind, np.arange())
right_channel_IR = r_eq_hrir_S.content_m(pos_ind, np.arange())
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

# ukazi.m:468 -- Note: See m:447
# ############## konvolucija v frekvenčni domeni #######################
# convert stereo to mono sound
posnetekMONO = (posnetek[:, 0] + posnetek[:, 1]) / 2

# left channel
X = np.fft.fft((posnetekMONO, np.concatenate(np.zeros(len(left_channel_IR) - 1, 1))))
Y = np.fft.fft((left_channel_IR.T, np.concatenate(np.zeros(len(posnetekMONO) - 1, 1))))
posnetek3D[:, 1] = np.fft.ifft(np.multiply(X, Y))

# left channel
X = np.fft.fft((posnetekMONO, np.concatenate(np.zeros(len(right_channel_IR) - 1, 1))))
Y = np.fft.fft((right_channel_IR.T, np.concatenate(np.zeros(len(posnetekMONO) - 1, 1))))
posnetek3D[:, 2] = np.fft.ifft(multiply(X, Y))
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

# ukazi.m:499
# --------------------------------------------------------------------
# AUDIO EFEKT DISTORTION
# (C) ROSIS, LSPO, FERI, Univerza v Mariboru, Slovenia
# --------------------------------------------------------------------

# ukazi.m:503 -- Preverjeno z Matlab
# ################### akustična kitara
Fs, y = wavfile.read('./2-Linearni Sistemi in Konvolucija/SotW.wav')
sd.play(y, Fs)

sd.stop()

# ukazi.m:511 -- Preverjeno z Matlab
# --------------------------------------------------------------------
# #################### efekt distortion ##################
#            y = vhodni signal
#            Fs = Vzorčevalna frekvenca

# ukazi.m:516 -- Note: Podatki, ki jih vrne wavfile niso normalizirani, medtem ko podatki, ki jih vrne Matlabov
# Audioread so, tako da je za enak A rezultat drugačen.
from functions import distortion   # Import custom function from functions.py in root folder.

A = 0.0002
dy = distortion(A, y)
sd.play(dy)

sd.stop()

# ukazi.m:526
# --------------------------------------------------------------------
# LINEARNI SISTEMI IN KONVOLUCIJA
# (C) ROSIS, LSPO, FERI, Univerza v Mariboru, Slovenia
#
# Poglavje 2: Slike (2D signali)
# --------------------------------------------------------------------
# SOBELOV OPERATOR

# ukazi.m:534 -- Note: Nejasnost v vrstici ~653
plt.close('all')
A = pylab.array(Image.open(Path('./2-Linearni Sistemi in Konvolucija/Valve_original.png')))
plt.figure()
plt.axis('off')
plt.imshow(A)
plt.title('originalna slika')

# ukazi.m:539
Sobel_x = np.outer([1, 2, 1], [+1, 0, -1])
Sobel_y = np.outer([+1, 0, -1], [1, 2, 1])

B_x = np.empty([len(A)+2, len(A[0, :])+2, 3])   # To match the size of convolution output.
B_x[:, :, 0] = conv2(A[:, :, 0], Sobel_x)
B_x[:, :, 1] = conv2(A[:, :, 1], Sobel_x)
B_x[:, :, 2] = conv2(A[:, :, 2], Sobel_x)

# Če porežemo vrednosti...
B_x = np.clip(B_x, 0, 255)

# B_x = uint8( (B_x - min(B_x(:))) / (max(B_x(:)) - min(B_x(:))) *255 );
# Verjetno ne razumem namena zgornje formule v Matlabu; meni vse vrednosti zaokroži na 0 in 255, je to željeno
# delovanje ali napaka pri zaokroževanju?
# Če skaliramo v razpon 0-255...
# B_x = (B_x - np.min(B_x)) / (np.max(B_x) - np.min(B_x)) * 255

B_x = B_x.astype(np.uint8)

plt.figure()
plt.axis('off')
plt.imshow(B_x)
plt.title('Sobelov operator v x smeri')

# ukazi.m:550
B_y = np.empty([len(A)+2, len(A[0, :])+2, 3])   # To match the size of convolution output.
B_y[:, :, 0] = conv2(A[:, :, 0], Sobel_y)
B_y[:, :, 1] = conv2(A[:, :, 1], Sobel_y)
B_y[:, :, 2] = conv2(A[:, :, 2], Sobel_y)

# Če porežemo vrednosti...
B_y = np.clip(B_y, 0, 255)
# Če skaliramo v razpon 0-255...
# B_y = (B_y - np.min(B_y)) / (np.max(B_y) - np.min(B_y)) * 255

B_y = B_y.astype(np.uint8)

plt.figure()
plt.axis('off')
plt.imshow(B_y)
plt.title('Sobelov operator v y smeri')

# ukazi.m:559
# --------------------------------------------------------------------
# SOBELOV OPERATOR - drugi???

# ukazi.m:562 -- Note: Preverjeno z Matlab
plt.close('all')
C = pylab.array(Image.open(Path('./2-Linearni Sistemi in Konvolucija/Bikesgray.jpg')))
plt.figure()
plt.axis('off')
plt.imshow(C, cmap='gray')
plt.title('Originalna slika')

Sobel_x = np.outer([1, 2, 1], [+1, 0, -1])
Sobel_y = np.outer([+1, 0, -1], [1, 2, 1])

# ukazi.m:570
D_x = conv2(C, Sobel_x)
plt.figure()
plt.axis('off')
plt.imshow(D_x, cmap='gray')
plt.title('Sobelov operator v x smeri')

D_y = conv2(C, Sobel_y)
plt.figure()
plt.axis('off')
plt.imshow(D_y, cmap='gray')
plt.title('Sobelov operator v y smeri')

plt.figure()
plt.axis('off')
plt.imshow(np.sqrt(D_x ** 2 + D_y ** 2), cmap='gray')
plt.title('Združen Sobelov operator v x in y smeri')

# ukazi.m:584
W_xy = np.sqrt(D_x ** 2 + D_y ** 2)
W_xy = (abs(255 - W_xy))
W_xy = W_xy.astype(np.uint8)
W_xy[W_xy > 140] = 255
plt.figure()
plt.axis('off')
plt.imshow(W_xy, cmap='gray')
plt.title('Združen Sobelov operator v x in y smeri')

# ukazi.m:590
# --------------------------------------------------------------------
# Laplacov operator

# ukazi.m:593 -- Note: Preverjeno z Matlab
plt.close('all')
C = pylab.array(Image.open(Path('./2-Linearni Sistemi in Konvolucija/Bikesgray.jpg')))
plt.figure()
plt.axis('off')
plt.imshow(C, cmap='gray')

Laplace = np.array(([0, - 1, 0], [- 1, 4, - 1], [0, - 1, 0]))

D_x = conv2(C, Laplace)
plt.figure()
plt.axis('off')
plt.imshow(D_x, cmap='gray')

# ukazi.m:602
# --------------------------------------------------------------------
# Glajenje slike - Gaussov operator

# ukazi.m:605 -- Note: Preverjeno z Matlab
plt.close('all')
C = pylab.array(Image.open(Path('./2-Linearni Sistemi in Konvolucija/Bikesgray.jpg')))
plt.figure()
plt.axis('off')
plt.imshow(C, cmap='gray')
plt.title('Originalna slika')

# sliko zgladimo z Gausovim operatorjem velikosti 5x5
Gauss = 1 / 159 * np.array(([2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]))
fig = plt.figure()
X = np.arange(Gauss[:, ].size)  # We have to create X and Y axis to feed the plot.
Y = np.arange(Gauss[0, :].size)
X, Y = np.meshgrid(X, Y)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Gauss, cmap='Set1')
ax.view_init(30, -135)          # Rotate it to same point of view as Matlab
plt.title('2D Gaussova funkcija')
plt.tight_layout()
plt.show()

# ukazi.m:615
D_x = conv2(C, Gauss)
plt.figure()
plt.axis('off')
plt.imshow(D_x, cmap='gray')
plt.title('zZglajena slika: Gauss 5x5')

# ukazi.m:619 -- Note: Preverjeno z Matbab
# ###### Poljubna velikost Gaussovega operatorja NxN
N = 25
# Matlab kot drugi parameter sprejme vrednost α, ki se v standardno deviacijo (σ), ki jo potrebuje
# scipy.signal.windows.gaussian pretvori po naslednji formuli: σ = (L – 1)/(2α), kjer je L dolžina okna.
w = gausswin(N, (N - 1) / 5)        # Privzeta alpha v matlabu je 2.5
Gauss = np.outer(w, w)
Gauss = Gauss / sum(w)
fig = plt.figure()
X = np.arange(Gauss[:, 0].size)      # We have to create X and Y axis to feed the plot.
Y = np.arange(Gauss[0, :].size)
X, Y = np.meshgrid(X, Y)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Gauss, cmap='Set1')
ax.view_init(30, -135)              # Rotate it to same point of view as Matlab
plt.title('2D Gaussova funkcija')
plt.tight_layout()
plt.show()

# ukazi.m:626
D_x = conv2(C, Gauss)
plt.figure()
plt.imshow(D_x, cmap='gray')
plt.axis('off')
plt.title('Zglajena slika: Gauss {0}x{0}'.format(N))
