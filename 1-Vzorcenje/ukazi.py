# Author: Martin Konečnik
# Contact: martin.konecnik@gmail.com
# Licenced under MIT

# 1-Vzorčenje
# Predlagan IDE PyCharm Community Edition (na voljo za Windows, macOS in Linux)
# https://www.jetbrains.com/pycharm/download
# Navodila za pridobitev potrebnih knjižnic:
# https://www.jetbrains.com/help/pycharm/installing-uninstalling-and-upgrading-packages.html
# Kratka navodila. Znotraj GUI knjižnice dodamo prek File->Settings->Project: Name->Project Interpreter.
# V tem oknu na desni strani kliknemo na plus in vpišemo ime knjižnice.
# Privzeta bližnjica za zagon izbora je Alt+Shift+E

import numpy as np
import scipy.signal
from matplotlib import cm  # color mapping
import pylab as pylab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sounddevice as sd
from pathlib import Path
from PIL import Image

# ----------------------------------------------------------------------------------------
# Vzorčenje in Nyquistov teorem;
Fvz = 100  # Frekvenca vzorčenja (v Hz)
T = 1  # dolžina signala (v s)
i = np.arange(float(T) * Fvz) / Fvz  # vektor časovnih indeksov
f1 = 5  # frekvenca sinusoide
A1 = 1  # amplituda sinusoide
faza1 = 0.0  # faza sinusoide

# ukazi.m:10 -- NOTE: Enak rezultat kot v Matlab 2017b
# izris sinuside pri razlicnih fazah
plt.figure()  # Ta vrstica ni nujna, če odpremo le eno okno.
for faza1 in np.arange(0, 6.1, 0.1):
    plt.cla()
    # pri množenju z matriko je potrebno uporabiti numpy.np.dot(...)
    s = np.dot(A1, np.sin(np.dot(2 * np.pi * f1, i) + faza1 * np.pi))
    plt.plot(i, s)
    setattr(plt.gca, 'YLim', [-1, 1])
    plt.title('Fvz = {0} Hz, Frekvenca = {1} Hz, faza = {2} $\pi$'.format(Fvz, f1, round(faza1, 1)))
    plt.xlabel('Čas (s)')
    plt.ylabel('Amplituda (dB)')
    plt.tight_layout()
    plt.waitforbuttonpress()

# ukazi.m:23 -- NOTE: Preverjeno z Matlab
# izris sinusid pri različnih frekvencah
faza1 = 0.0
plt.figure()
for f1 in np.arange(Fvz + 1):
    plt.cla()  # počistimo graf za naslednjo iteracijo
    s = np.dot(A1, np.sin(np.dot(2 * np.pi * f1, i) + faza1 * np.pi))
    plt.plot(i, s)
    plt.ylim(-1, 1)
    plt.title('Fvz = {0} Hz, Frekvenca = {1} Hz, faza = {2} $\pi$'.format(Fvz, f1, faza1))
    plt.xlabel('Čas (s)')
    plt.ylabel('Amplituda (dB)')
    plt.pause(0.025)

# ukazi.m:37 -- NOTE: Preverjeno z Matlab
# izris sinusoid s frekvenco f1 in Fvz-f1
f1 = 1
s1 = np.sin(np.dot(2 * np.pi * f1, i) + faza1 * np.pi)
plt.figure()
plt.plot(i, s1, 'b')
f2 = Fvz - f1
s2 = np.sin(np.dot(np.dot(np.dot(2, np.pi), f2), i) + np.dot(faza1, np.pi))
plt.plot(i, s2, 'r')
plt.xlabel('Čas (s)')
plt.ylabel('Amplituda')
plt.title('Fvz = {0} Hz, Frekvenca1 = {1} Hz, Frekvenca2 = {2} Hz, faza = {3} $\pi$'.format(Fvz, f1, f2, faza1))
plt.tight_layout()
# ----------------------------------------------------------------------------------------

# glej tudi primera v Mathematici: (WheelIllusion.nbp in SamplingTheorem.nbp)

# V Z O R Č E N J E    Z V O K A
# -----------------------------------------------------------------------------------------
# ukazi.m:56 -- NOTE: Primerljiv rezultat v Matlab
# vzorčenje zvoka
Fs = 44100  # vzorčevalna frekvenca
bits = 16  # bitna ločljivost
nchans = 1  # 1 (mono), 2 (stereo).
posnetek = sd.rec(5 * Fs, Fs, nchans, blocking=True)

plt.figure()
plt.plot(posnetek)

sd.play(posnetek, 44100)
sd.play(posnetek, 44100 / 2)
sd.play(posnetek, 2 * 44100)

# -----------------------------------------------------------------------------------------
# ukazi.m:73 -- NOTE: Primerljiv rezultat v Matlab
# ali zaznate fazne spremembe? Spreminjajte faza1 med 0 in 2.0 in poženite ta demo...
Fvz = 44100  # vzorčevalna frekvenca
T = 3  # čas v sekundah
i = np.arange(0.0, T * Fvz, 1) / Fvz  # vektor časovnih indeksov
f1 = 500  # frekvenca sinusoide
A1 = 0.3  # amplituda sinusoide
faza1 = 1.0  # faza sinusoide

s = np.dot(A1, np.sin(np.dot(2 * np.pi * f1, i) + faza1 * np.pi))  # tvorjenje sinusoide
s2 = np.dot(A1, np.sin(np.dot(2 * np.pi * f1, i) + 0 * np.pi))  # tvorjenje sinusoide

sd.play(np.concatenate((s, s2)), Fvz)  # pozor, dvojni oklepaji pri concatenate, ker sta s1 in s2 en parameter!
# "navadna" polja združiš z s+s2, v numpy.array pa to sešteje istoležne elemente


# -----------------------------------------------------------------------------------------
# ukazi.m:88 -- NOTE: Primerljiv rezultat v Matlab
# trije poskusi: 1. f1 = 50;
#                2. f1 = 450;
#                3. f1 = 1450;
#                4. f1 = 2450;

Fvz = 44100  # vzorčevalna frekvenca
T = 3  # čas v sekundah
i = np.arange(T * Fvz) / Fvz  # vektor časovnih indeksov
f1 = 50  # frekvenca sinusoide
A1 = 5.5  # amplituda sinusoide
faza1 = 0.0  # faza sinusoide
f2 = f1 + 1  # frekvenca druge sinusoide

s1 = np.dot(A1, np.sin(np.dot(2 * np.pi * f1, i) + faza1 * np.pi))  # tvorjenje prve sinusoide
s2 = np.dot(A1, np.sin(np.dot(2 * np.pi * f2, i) + faza1 * np.pi))  # tvorjenje druge sinusoide
sd.play(np.concatenate((s1, s2)), Fvz)

# -----------------------------------------------------------------------------------------
# ukazi.m:106 -- NOTE: Primerljiv rezultat v Matlab
# ali zaznate zvok netopirja pri 90000 Hz? Nyquist?
Fvz = 44100  # vzorčevalna frekvenca
T = 3
i = np.arange(T * Fvz) / Fvz  # vektor časovnih indeksov
fnetopir = 140000  # frekvenca sinusoide
A1 = 5.5  # amplituda sinusoide
faza1 = 1.0  # faza sinusoide

s = np.dot(A1, np.sin(np.dot(2 * np.pi * fnetopir, i) + faza1 * np.pi))  # tvorjenje sinusoide
sd.play(s, Fvz)

# ukazi.m:118 -- NOTE: Preverjeno z Matlab
# izris sinuside pri razlicnih fazah (verzija 2)
Fvz = 100
T = 1
i = np.arange(T * Fvz) / Fvz
f1 = 5
A1 = 5
faza1 = 0.0

# spremninjanje frekvence...
plt.close('all')
fig, ax = plt.subplots(2)  # create a figure with 2 subplots
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

for f1 in np.arange(0, Fvz+1):
    ax[0].clear()
    ax[1].clear()

    s = np.sin(np.dot(2 * np.pi * f1, i) + faza1 * np.pi)
    ax[0].plot(s)
    ax[0].set_ylim(-1, 1)
    ax[0].set_title('Časovna domena: Fvz = {0} Hz, Frekvenca = {1} Hz, faza = {2} $\pi$'.format(Fvz, f1, faza1))

    ax[1].plot(abs(np.fft.fft(s)), 'r')
    ax[1].set_ylim(-1, 1)
    ax[1].set_title('Frekvenčna domena (abs): Fvz = {0} Hz, Frekvenca = {1} Hz, faza = {2} $\pi$'.format(Fvz, f1, faza1))

    plt.waitforbuttonpress()


# in faze... (več o tem na naslednjih vajah)
f1 = 5
A1 = 5
faza1 = 0.0
plt.close('all')
fig, ax = plt.subplots(2)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

for faza1 in np.arange(0, 2.1, 0.1):
    ax[0].clear()
    ax[1].clear()

    s = np.sin(np.dot(2 * np.pi * f1, i) + faza1 * np.pi)
    ax[0].plot(s)
    ax[0].set_ylim(-1, 1)
    ax[0].set_title('Časovna domena: Fvz = {0} Hz, Frekvenca = {1} Hz, faza = {2} $\pi$'.format(Fvz, f1, round(faza1, 1)))

    ax[1].plot(abs(np.fft.fft(s)), 'r')
    ax[1].set_ylim(-1, 1)
    ax[1].set_title('Frekvenčna domena (abs): Fvz = {0} Hz, Frekvenca = {1} Hz, faza = {2} $\pi$'.format(Fvz, f1, round(faza1, 1)))

    plt.waitforbuttonpress()

# S L I K E
# -----------------------------------------------------------------------------------------
# ukazi.m:181 -- NOTE: Preverjetno v Matlabu. Del ne deluje pravilno (označeno)
# vzorčenje slik in Moire
# Če datoteke ne najde, preverite pod "Settings -> Project: Name -> Project Structure" kje je root.
A = pylab.array(Image.open(Path('./1-Vzorcenje/Moire.jpg')))
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(A)
plt.title('originalna slika')

pvz = 3
plt.figure()
plt.axis('off')
plt.imshow(A[::pvz, ::pvz])
plt.title('podvzorčena slika: faktor podvzorčenja {0}'.format(pvz))

# Celotna slika bistveno svetlejša kot v matlabu. Pri bitni ločljivosti 2, sta namesto sivin rumena in modra barva.
# Primerjaj podatke.
st_bit = 2
kvant = 2 ** (9 - st_bit)
plt.figure(figsize=(10, 10))
plt.imshow(np.dot(np.round(A[:, :, :] / kvant), kvant))
plt.title('slika pri bitni ločljivosti {0}'.format(st_bit))

fig, ax = plt.subplots(2, 2)
fig.tight_layout()

ax[0, 0].imshow(np.dot(np.round(A[:, :, :] / kvant), kvant))
ax[0, 0].set_title('slika pri bitni ločljivosti {0}'.format(st_bit))

ax[0, 1].imshow(np.dot(np.round(A[:, :, 0] / kvant), kvant))
ax[0, 1].set_title('ravnina R pri bitni ločljivosti {0}'.format(st_bit))

ax[1, 0].imshow(np.dot(np.round(A[:, :, 1] / kvant), kvant))
ax[1, 0].set_title('ravnina G pri bitni ločljivosti {0}'.format(st_bit))

ax[1, 1].imshow(np.dot(np.round(A[:, :, 2] / kvant), kvant))
ax[1, 1].set_title('ravnina B pri bitni ločljivosti {0}'.format(st_bit))

# -----------------------------------------------------------------------------------------
# ukazi.m:215
# spekter slik, Moire in Diskretna Fourierova transformacija (fft2)
A = plt.imread(os.path.join(os.path.curdir, 'Moire.jpg'))
plt.figure()
plt.imshow(A)
plt.title('originalna slika')
plt.close('all')

B = np.double(A[:, :, 1])
X, Y = np.meshgrid(np.arange(100), np.arange(100))
Z = abs(np.fft.fft2(B - np.mean(np.ravel(B)))[0:100, 0:100])
colors = cm.Blues(Z)
rcount, ccount, _ = colors.shape

fig = plt.figure()
fig.set_size_inches(7, 7)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
surf.set_facecolor((0, 0, 0, 0))
plt.title('R ravnina')
plt.show()

# ukazi.m:226
B = np.double(A[:, :, 2])
X, Y = np.meshgrid(np.arange(100), np.arange(100))
Z = abs(np.fft.fft2(B - np.mean(np.ravel(B)))[0:100, 0:100])
colors = cm.Blues(Z)
rcount, ccount, _ = colors.shape

fig = plt.figure()
fig.set_size_inches(7, 7)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
surf.set_facecolor((0, 0, 0, 0))
plt.title('G ravnina')
plt.show()

# ukazi.m:228
B = np.double(A[:, :, 3])
X, Y = np.meshgrid(np.arange(100), np.arange(100))
Z = abs(np.fft.fft2(B - np.mean(np.ravel(B)))[0:100, 0:100])
colors = cm.Blues(Z)
rcount, ccount, _ = colors.shape

fig = plt.figure()
fig.set_size_inches(7, 7)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
surf.set_facecolor((0, 0, 0, 0))
plt.title('B ravnina')
plt.show()

# ukazi.m:231
# prevzorčena slika..................................
pvz = 4
B = np.double(A[0::pvz, 0::pvz, 1])
X, Y = np.meshgrid(np.arange(100), np.arange(100))
Z = abs(np.fft.fft2(B - np.mean(np.ravel(B)))[0:100, 0:100])
colors = cm.Blues(Z)
rcount, ccount, _ = colors.shape

fig = plt.figure()
fig.set_size_inches(7, 7)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
surf.set_facecolor((0, 0, 0, 0))
plt.title('R ravnina, po podvzorenju s faktorjem {0}'.format(pvz))
plt.show()

# ukazi.m:235
B = np.double(A[0::pvz, 0::pvz, 2])
X, Y = np.meshgrid(np.arange(100), np.arange(100))
Z = abs(np.fft.fft2(B - np.mean(np.ravel(B)))[0:100, 0:100])
colors = cm.Blues(Z)
rcount, ccount, _ = colors.shape

fig = plt.figure()
fig.set_size_inches(7, 7)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
surf.set_facecolor((0, 0, 0, 0))
plt.title('G ravnina, po podvzorenju s faktorjem {0}'.format(pvz))
plt.show()

# ukazi.m:237
B = np.double(A[0::pvz, 0::pvz, 3])
X, Y = np.meshgrid(np.arange(100), np.arange(100))
Z = abs(np.fft.fft2(B - np.mean(np.ravel(B)))[0:100, 0:100])
colors = cm.Blues(Z)
rcount, ccount, _ = colors.shape

fig = plt.figure()
fig.set_size_inches(7, 7)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
surf.set_facecolor((0, 0, 0, 0))
plt.title('B ravnina, po podvzorenju s faktorjem {0}'.format(pvz))
plt.show()

# -----------------------------------------------------------------------------------------
# ukazi.m:241
# podvzorenje slik in operator povpreenja

A = plt.imread(os.path.join(os.path.curdir, 'Moire.jpg'))
plt.figure()
plt.imshow(A)
plt.title('originalna slika')

pvz = 3  # faktor podvzorčenja
plt.figure()
plt.imshow(A[0::pvz, 0::pvz, :])
plt.title('podvzorčena slika: faktor podvzorčenja {0}'.format(pvz))

# ukazi.m:253
# operator povprečenja (verzija 1)
# Mislim, da ne dela kot bi moralo. Could probably benefit from refactoring.
D = 3  # premer lokalne okolice piksla, na kateri se izračuna povprečna vrednost
B = np.ndarray((A.shape[0] - D + 1, A.shape[1] - D + 1, D))
for r in np.arange(0, A.shape[0] - D).reshape(-1):
    for c in np.arange(0, A.shape[1] - D).reshape(-1):
        C = A[r + np.arange(0, D - 1), c + np.arange(0, D - 1), 0]
        B[r, c, 0] = np.mean(np.ravel(C))
        C = A[r + np.arange(0, D - 1), c + np.arange(0, D - 1), 1]
        B[r, c, 1] = np.mean(np.ravel(C))
        C = A[r + np.arange(0, D - 1), c + np.arange(0, D - 1), 2]
        B[r, c, 2] = np.mean(np.ravel(C))

plt.figure()
plt.imshow(np.uint8(B))
plt.title('zglajena slika')

# ukazi.m:270
# operator povpreenja (verzija 2)
# isti operator povprečenja kot zgoraj, implementiran nekoliko drugače (veliko hitreja izvedba)
D = 3
B = np.ndarray((A.shape[0] + D - 1, A.shape[1] + D - 1, D))
B[:, :, 0] = scipy.signal.convolve2d(np.double(A[:, :, 0]), np.ones((D, D), np.float) / D ** 2)
B[:, :, 1] = scipy.signal.convolve2d(np.double(A[:, :, 1]), np.ones((D, D), np.float) / D ** 2)
B[:, :, 2] = scipy.signal.convolve2d(np.double(A[:, :, 2]), np.ones((D, D), np.float) / D ** 2)
B = np.uint8(B)
plt.figure()
plt.imshow(B)
plt.title('zglajena slika')

# prikaz
pvz = 3  # faktor podvzorčenja
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(A[0::pvz, 0::pvz, :])
plt.title('podvzorčena slika: faktor podvzorčenja {0}'.format(pvz))
plt.subplot(1, 2, 2)
plt.imshow(B[1::pvz, 1::pvz, :])
plt.title('zglajena podvzorena slika: faktor podvzorenja {0}'.format(pvz))
