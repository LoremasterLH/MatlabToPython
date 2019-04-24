# Author: Martin Konečnik
# Contact: martin.konecnik@gmail.com
# Licenced under MIT

# 3-Fourirjeva Analiza
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

# ukazi.m:1
# -------------------------------------------------------------
# Skalarni produkt dveh sinusoid

# ukazi.m:3 -- Note: Preverjeno z Matlab
plt.close('all')
Fvz = 512  # frekvenca vzorčenja
T = 1  # dolžina signala (v sekundah)
i = np.arange(T * Fvz) / Fvz  # časovni trenutki vzorčenja
f1 = 6  # frekvenca prve sinusoide (v Hz) - igrajte se z njeno vrednostjo in jo spreminjajte po korakih 0.1 Hz (med 5 Hz in 7 Hz)
f2 = 5  # frekvenca druge sinusoide (v Hz)
s1 = np.sin(2 * np.pi * f1 * i)
s2 = np.sin(2 * np.pi * f2 * i)
f, ax = plt.subplots(2)
ax[0].plot(i.tolist(), s1.tolist())
ax[0].set_title('{0} Hz'.format(f1))
ax[1].plot(i.tolist(), s2.tolist(), 'r')
ax[1].set_title('{0} Hz'.format(f2))
ax[1].set_xlabel('čas (s)')
ax[1].set_ylabel('amplituda')
plt.tight_layout()

skalarni_produkt = np.dot(s1, s2)

# ukazi.m:19 -- Note: Rezultat skalarnega produkta se ne sklada z rezultatom v Matlabom.
# -------------------------------------------------------------
# skalarni produkt dveh RAZLIČNIH sinusoid pri razlicnih fazah

# ukazi.m:21
plt.close('all')
Fvz = 512  # frekvenca vzorčenja
T = 1  # dolžina signala (v sekundah)
i = np.arange(T * Fvz) / Fvz  # časovni trenutki vzorčenja
f1 = 6  # frekvenca prve sinusoide (v Hz)
f2 = 5  # frekvenca druge sinusoide (v Hz)
s1 = np.sin(2 * np.pi * f1 * i)
skalarni_produkt = np.empty(np.arange(-1, 1.05, 0.05).size)  # Dolžina zanke.

fig, ax = plt.subplots(2)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
ax[0].plot(i, s1)
ax[0].set_title('{0} Hz'.format(f1))
for faza in np.arange(-1, 1.05, 0.05):  # fazo povečujemo v koraku 0.05pi, od -pi do +pi
    s2 = np.sin(2 * np.pi * f2 * i + faza * np.pi)
    ax[1].clear()
    ax[1].plot(i, s2, 'r')
    ax[1].set_title('{0} Hz, faza = {1}*$\pi$'.format(f2, round(faza, 2)))
    skalarni_produkt[int(faza * 20 + 20)] = np.dot(s1, s2)  # shranjujemo vrednost skalarnega produkta
    plt.waitforbuttonpress()

plt.figure()
plt.plot(np.arange(-1, 1.05, 0.05), skalarni_produkt)  # izrišemo vrednost skalarnega produkta pri razlicnih fazah...
plt.xlabel('faza ($\pi$)')

# ukazi.m:39 -- Note: Preverjeno z Matlab
# -------------------------------------------------------------
# skalarni produkt ob frekvenčnem neujemanju - spektralno prepuscanje (Spectral leakage)
# vec o tem v spodnjih zgledih - razlaga na voljo tudi na: http://www.dsptutor.freeuk.com/analyser/guidance.html#leakage

# ukazi.m:59
plt.close('all')
Fvz = 512  # frekvenca vzorčenja
T = 2  # dolžina signala (v sekundah)
i = np.arange(T * Fvz) / Fvz  # časovni trenutki vzorčenja
f1 = 5
faza1 = 0  # rand
f2 = 5
faza2 = 0  # rand

s1 = np.sin(2 * np.pi * f1 * i + faza1 * np.pi)
skalarni_produkt = np.empty(np.arange(- 1, 1.001, 0.001).size)
for dfrek in np.arange(-1, 1.001, 0.001):  # frekvenčno neujemanje dfrek povecujemo v koraku 0.001 Hz, od -1 Hz do + 1 Hz
    print(dfrek)
    s2 = np.sin(2 * np.pi * (f2 + dfrek) * i + faza2 * np.pi)
    skalarni_produkt[int(dfrek * 1000 + 1000)] = np.dot(s1, s2)

plt.figure()
plt.plot(np.arange(-1, 1.001, 0.001), skalarni_produkt)
plt.xlabel('frekvenčno neujemanje (Hz)')

plt.plot([- 1, 1], [0, 0], 'r')

# ukazi.m:60 -- Note: Preverjeno z Matlab - drugačna implementacija grafa kot navadno
# -------------------------------------------------------------
# skalarni produkt dveh sinusoid z ENAKIMA FREKVENCAMA IN RAZLIČNIMA FAZAMA

# ukazi.m:62
Fvz = 512
T = 1
i = np.arange(T * Fvz) / Fvz  # vektor časovnih indeksov
f1 = 5
A1 = 1
f2 = 5
faza2 = 0

plt.figure()
for faza1 in np.arange(- 1, 1.05, 0.05):  # fazo povečujemo v koraku 0.05pi, od -pi do +pi
    plt.clf()

    # ustvarimo signal
    x = np.dot(A1, np.sin(2 * np.pi * f1 * i + faza1 * np.pi))
    # ustvarimo sinusoido
    s1 = np.dot(A1, np.sin(2 * np.pi * f2 * i + faza2 * np.pi))
    Xs1 = np.dot(s1, x)  # skalarni produkt s sinusoido

    plt.subplot(2, 1, 1)
    plt.plot(i, x)      # narišemo prvo
    plt.plot(i, s1, 'r')  # in še drugo sinusoido
    plt.xlabel('čas (s)')
    plt.ylabel('Amplituda')
    plt.title('frekvenca = {0}, faza = {1} $\pi$'.format(f1, round(faza1, 2)))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.subplot(2, 1, 2)
    plt.plot([0, 0], [0, Xs1], lineWidth=2)  # narišemo amplitude kosinusov
    plt.xlabel('realna os')
    plt.ylabel('imaginarna os')
    plt.xlim(-300, 300)
    plt.ylim(-300, 300)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('frekvenca sin = {0}, faza2 = {1} $\pi$'.format(f2, faza2))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.pause(0.5)

# ukazi.m:95 -- Note: Preverjeno z Matlab
# -------------------------------------------------------------
# skalarni produkt dveh sinusoid z ENAKIMA FREKVENCAMA IN RAZLIČNIMA FAZAMA
# analiza s sinusoidami in kosinusoidami

# ukazi.m:98
plt.close('all')
Fvz = 512
T = 1
i = np.arange(T * Fvz) / Fvz  # vektor časovnih indeksov
f1 = 5
A1 = 1
f2 = 5
faza2 = 0

plt.figure()
for faza1 in np.arange(- 1, 1.05, 0.05):
    plt.clf()
    # ustvarimo signal
    x = A1 * np.sin(2 * np.pi * f1 * i + faza1 * np.pi)
    # ustvarimo sinusoido
    s1 = 1 * np.sin(2 * np.pi * f2 * i + faza2 * np.pi)
    # ustvarimo kosinusoido
    c1 = 1 * np.cos(2 * np.pi * f2 * i + faza2 * np.pi)

    Xs1 = np.dot(s1, x)  # skalarni produkt s sinusoido
    Xc1 = np.dot(c1, x)  # skalarni produkt s kosinusoido

    plt.subplot(2, 1, 1)
    plt.plot(i, x)      # narišemo prvo
    plt.plot(i, s1, 'r')  # in pe drugo sinusoido
    plt.xlabel('čas (s)')
    plt.ylabel('Amplituda')
    plt.title('frekvenca = {0}, faza = {1} $\pi$'.format(f1, round(faza1, 2)))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.subplot(2, 1, 2)
    X = np.fft.fft(x)  # FFT!!
    plt.plot([0, Xc1], [0, Xs1], lineWidth=2)  # narišemo amplitudo kosinusov
    plt.xlabel('x os')
    plt.ylabel('y os')
    plt.ylim(-300, 300)
    plt.xlim(-300, 300)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('frekvenca sin in cos = {0}, faza2 = {1} $\pi$'.format(f2, faza2))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.pause(0.5)

# ukazi.m:136 -- Note: Preverjeno z Matlab
# -------------------------------------------------------------
# skalarni produkt dveh sinusoid z ENAKIMA FREKVENCAMA IN RAZLIČNIMA FAZAMA
# komplesna analiza s fft

# ukazi.m:1391
Fvz = 512
T = 1
i = np.arange(T * Fvz) / Fvz  # vektor časovnih indeksov
f1 = 5
A1 = 1
faza1 = 0.5

plt.figure()
for faza1 in np.arange(-1, 1.05, 0.05):
    plt.clf()
    # ustvarimo sinusiudo
    x = A1 * np.sin(2 * np.pi * f1 * i + faza1 * np.pi)

    plt.subplot(2, 1, 1)
    plt.plot(i, x)      # narišemo prvo
    plt.plot(i, s1, 'r')  # in se drugo sinusoido
    plt.xlabel('čas (s)')
    plt.ylabel('Amplituda')
    plt.title('faza = {0} $\pi$'.format(round(faza1, 2)))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.subplot(2, 1, 2)
    X = np.fft.fft(x)  # FFT!!
    plt.plot([0, np.real(X[f1])], [0, np.imag(X[f1])], lineWidth=2)  # narišemo amplitude kosinusov
    plt.xlabel('realna os')
    plt.ylabel('imaginarna os')
    plt.ylim(-300, 300)
    plt.xlim(-300, 300)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.pause(0.5)

# ukazi.m:167 -- Note: Preverjeno z Matlab vendar deluje počasi
# #####################################################
# skalarni produkt dveh sinusoid z ENAKIMA FREKVENCAMA IN RAZLIČNIMA FAZAMA
# komplesna analiza s fft - pogled z druge perspektive (realni in imaginarni del fft-ja)

# ukazi.m:171
Fvz = 512
T = 1
i = np.arange(T * Fvz) / Fvz  # vektor časovnih indeksov
f1 = 5
A1 = 5
faza1 = 0.5

plt.figure()
for faza1 in np.arange(- 1, 1.05, 0.05):
    plt.clf()

    # ustvarimo sinusiudo
    x = A1 * np.sin(2 * np.pi * f1 * i + faza1 * np.pi)
    X = np.fft.fft(x)  # FFT!!

    plt.subplot(2, 1, 1)
    plt.stem(Fvz * i / T, np.real(X))  # narišemo amplitude kosinusov
    plt.xlabel('Frekvenca (Hz)')
    plt.ylabel('Amplituda')
    plt.title('realni del fft; faza = {0} $\pi$'.format(round(faza1, 2)))
    plt.ylim(-1500, 1500)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.subplot(2, 1, 2)
    plt.stem(Fvz * i / T, np.imag(X))  # narišemo amplitude sinusov
    plt.xlabel('Frekvenca (Hz)')
    plt.ylabel('Amplituda')
    plt.title('imaginarni del fft')
    plt.ylim(-1500, 1500)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.pause(0.05)

# ukazi.m:197 -- Note: Preverjeno z Matlab
# -------------------------------------------------------------
# superpozicija sinusoid, cosinusiode in sinusoide

# ukazi.m:200
Fvz = 512
T = 1
i = np.arange(T * Fvz) / Fvz  # vektor časovnih indeksov
f1 = 5
A1 = 5
faza1 = 0.5
f2 = 32
A2 = 3
faza2 = 0.3

# ustvarimo superpozicijo sinusoid
x = A1 * np.sin(2 * np.pi * f1 * i + faza1 * np.pi) + A2 * np.sin(2 * np.pi * f2 * i + faza2 * np.pi)
plt.figure()
plt.plot(i, x)  # narišemo signal

plt.xlabel('cas (s)')
plt.ylabel('Amplituda')
X = np.fft.fft(x)

plt.figure()
plt.subplot(2, 1, 1)
plt.stem(Fvz * i / T, np.real(X))
plt.xlabel('Frekvenca (Hz)')
plt.ylabel('Amplituda')
plt.title('realni del fft')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.subplot(2, 1, 2)
plt.stem(Fvz * i / T, np.imag(X))
plt.xlabel('Frekvenca (Hz)')
plt.ylabel('Amplituda')
plt.title('imaginarni del fft')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# ukazi.m:225 -- Note: Preverjeno z Matlab
# ###############################################################

plt.close('all')
plt.figure()
M = abs(X)  # dobimo amplitude
P = np.arctan2(np.imag(X), np.real(X))  # dobimo faze
plt.stem(Fvz * i / T, M)
plt.xlabel('Frekvenca')
plt.ylabel('Amplituda')
plt.title('Amplituda fft-ja')

# niso vse faze zanimive...
plt.figure()
plt.plot(P)
plt.title('Faza fft-ja')

# le faze tistih sinusoid, ki so dejansko prisotne v signalu
plt.figure()
P1 = np.multiply(P, (M > 0.1))
P1 = P1.squeeze()
plt.stem(Fvz * i / T, P1)
plt.xlabel('Frekvenca')
plt.ylabel('Faza')
plt.title('Faza fft-ja')

# POJASNILO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# atan2(imag(X),real(X)) vrne faze za enacbo M*sin(x-pi/2+theta), torej moramo ocenjeni fazi prišteti še pi/2;
#      P = P + pi/2;
# Ne pozabimo tudi, da smo pri generiranju signala x spremenjivko faza2 pomnožili s pi, torej je faza izražena v enotah pi
# KONEC POJASNILA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# -----------------------------------------------------------------------------------------------
################################################################################################

#             L A S T N O S T I   D F T

################################################################################################
# -----------------------------------------------------------------------------------------------

# ukazi.m:261 -- Note: Imaginarni deli so 0
# linearnost DFT-ja
x = np.random.randn(128, 1)
X = np.fft.fft(x)
y = np.random.randn(128, 1)
Y = np.fft.fft(y)
a = np.random.rand(1)
b = np.random.rand(1)
z = a * x + b * y
Z = np.fft.fft(z)

# graficni preizkus linearnosti DFT-ja
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(np.real(Z), lineWidth=2)
plt.plot(np.real(a * X + b * Y), 'r', lineWidth=1)
plt.xlabel('frekvenca')
plt.ylabel('realni del')
plt.title('fft.fft(a*x + b*y) in a*fft.fft(x) + b*fft.fft(y): primerjava realnih delov')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.subplot(2, 1, 2)
plt.plot(np.imag(Z), lineWidth=2)
plt.plot(np.imag(a * X + b * Y), 'r', lineWidth=1)
plt.ylabel('imaginarni del')
plt.xlabel('frekvenca')
plt.title('fft.fft(a*x + b*y) in a*fft.fft(x) + b*fft.fft(y): primerjava imaginarnih delov')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# ukazi.m:290 -- Note: Energija v frekvenčni domeni ni pravilna.
# -----------------------------------------------------------------------------------------------
# Parsevalov teorem
x = np.random.randn(128, 1)  # ustvarim signal
X = np.fft.fft(x)  # in njegovo frekvenčno transformiranko
energija_v_t_domeni = np.dot(x.T, x).item()
energija_v_f_domeni = np.mean(abs(X) ** 2)

# ukazi.m: -- Note: Preverjeno z Matlab
# -----------------------------------------------------------------------------------------------'
# fft in konvoluciija

# ukazi.m:300
print("\n" * 80)  # clc
# exit()  # za izbris spremenljivk, ponovno je potrebno importati knjižnice

Fs = 44100  # vzorčevalna frekvenca
bits = 16   # bitna ločljivost - ni uporabljena
T1 = 3
T2 = 2      # dolžina signalov

print('Snemam prvi posnetek v dolzini {0}s...'.format(T1))
posnetek1 = sd.rec(T1 * Fs, Fs, 1, blocking=True)

print('Snemam prvi posnetek v dolzini {0}s...'.format(T2))
posnetek2 = sd.rec(T2 * Fs, Fs, 1, blocking=True)

print('Predvajam prvi posnetek.')
sd.play(posnetek1, Fs, blocking=True)

print('Predvajam drugi posnetek.')
sd.play(posnetek2, Fs, blocking=True)

# ukazi.m:324
print("\n" * 80)  # clc
print('Konvolucija posnetkov v časovnem prostoru.')
efekt1 = np.empty([len(posnetek1) + len(posnetek2) - 1, 1])
tic = time.time()
efekt1[:, 0] = np.convolve(posnetek1[:, 0], posnetek2[:, 0])

toc = time.time() - tic
efekt1 = np.dot(efekt1 / max(efekt1), max(posnetek1))

print('Potrebovali smo "samo" {0} s'.format(round(toc * 100) / 100))
print('... in rezultat se sliši takole:')
sd.play(efekt1, Fs, blocking=True)

# ukazi.m:338
print('konvolucija posnetkov v frekvenčnem prostoru')
efekt2 = np.empty([len(posnetek1) + len(posnetek2) - 1, 1])
tic = time.time()
X = np.fft.fft(np.concatenate((posnetek1[:, 0], np.zeros(len(posnetek2[:, 0]) - 1))))
Y = np.fft.fft(np.concatenate((posnetek2[:, 0], np.zeros(len(posnetek1[:, 0]) - 1))))
efekt2[:, 0] = np.fft.ifft(np.multiply(X, Y))
toc = time.time()
efekt2 = np.dot(efekt2 / max(efekt2), max(posnetek1))

print('Potrebovali smo {0} s'.format(round(toc * 100) / 100))
print('... in rezultat se sliši takole:')
sd.play(efekt2, Fs, blocking=True)

# ukazi.m:352 -- Note: Odziv ne zgleda ustrezen.
# -----------------------------------------------------------------------------------------------'
# fft vsebina zvocnega posnetka

# ukazi.m:355
print("\n" * 80)  # clc
# exit()  # za izbris spremenljivk, ponovno je potrebno importati knjižnice

Fs = 44100
bits = 16
T1 = 5

print('Snemam prvi posnetek v dolzini {0} s...'.format(T1))

posnetek = sd.rec(T1 * Fs, Fs, 1, blocking=True)

plt.figure()
plt.plot(posnetek, lineWidth=0.4)
X = np.fft.fft(posnetek)

N = len(posnetek)
dF = Fs / N

f = np.concatenate((np.arange(dF, Fs / 2 + dF, dF), np.arange(Fs / 2 - dF, -dF, -dF)))
# ukazi3.m:363
plt.figure()
plt.plot(f, abs(X) / N, lineWidth=0.4)
plt.xlabel('Frekvenca (H)')
plt.title('Odziv magnitude')

#tmp = np.arange(dF, Fs / 2 + 0.2, 0.2)0
#tmp2 = np.arange(Fs / 2 - dF, -0.2, -0.2)