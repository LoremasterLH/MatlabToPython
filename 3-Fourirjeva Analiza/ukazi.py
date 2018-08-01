# 3-Fourirjeva Analiza
import numpy as np
import scipy.signal
from matplotlib import cm  # color mapping
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.plotly as py
import plotly.graph_objs as go
from PIL import Image

'-------------------------------------------------------------\n% skalarni produkt dveh sinusoid\n'
plt.close('all')

# ukazi.m:4
Fvz = 512
T = 1
i = np.arange(float(T)*Fvz)/Fvz
# i = smop.cat(arange(0.0, T * Fvz)) / Fvz
f1 = 6
f2 = 5
s1 = np.sin(np.dot(2 * np.pi * f1, i))
s2 = np.sin(np.dot(2 * np.pi * f2, i))
f, ax = plt.subplots(2)
ax[0].plot(i.tolist(), s1.tolist())
ax[0].set_title('{0} Hz'.format(f1))
ax[1].plot(i.tolist(), s2.tolist(), 'r')
ax[1].set_title('{0} Hz'.format(f2))
ax[1].set_xlabel('čas (s)')
ax[1].set_ylabel('amplituda')
plt.tight_layout()

skalarni_produkt = np.dot(s1, s2.T)

'-------------------------------------------------------------\n% skalarni produkt dveh RAZLIcNIH sinusoid pri razlicnih fazah\n'
plt.close('all')

# ukazi.m:22
Fvz = 512
T = 1
i = np.arange(float(T)*Fvz)/Fvz
f1 = 5
f2 = 6
s1 = np.sin(np.dot(2 * pi * f1, i))
skalarni_produkt = smop.matlabarray([])
plt.figure()
for faza in arange(- 1, 1.05, 0.05).reshape(-1):
    s2 = sin(dot(2 * pi * f2, i) + dot(faza, pi))
    plt.subplot(2, 1, 1)
    plt.plot(i.tolist(), s1.list())
    plt.title(smop.cat(str(f1), ' Hz'))
    plt.subplot(2, 1, 2)
    plt.plot(i.tolist(), s2.tolist(), 'r')
    plt.title(smop.cat(str(f2), ' Hz, faza = ', str(faza), '*pi'))
    plt.pause()
    skalarni_produkt[skalarni_produkt.shape[0] + 1] = dot(s1, s2.T)

plt.figure()
plt.plot(arange(- 1, 1.05, 0.05), skalarni_produkt)
plt.xlabel('faza (/pi)')

'-------------------------------------------------------------\n% skalarni produkt ob frekvencnem neujemanju - spektralno prepuscanje (Spectral leakage)\n% vec o tem v spodnjih zgledih - razlaga na voljo tudi na: http://www.dsptutor.freeuk.com/analyser/guidance.html#leakage\n% \n'

plt.close('all')
Fvz = 512
# ukazi3.m:40

T = 2
# ukazi3.m:41

i = smop.cat(arange(0.0, T * Fvz)) / Fvz
# ukazi3.m:42

f1 = 5
# ukazi3.m:43
faza1 = 0
# ukazi3.m:43

f2 = 5
# ukazi3.m:44
faza2 = 0
# ukazi3.m:44

s1 = sin(dot(2 * pi * f1, i) + faza1 * pi)
# ukazi3.m:45
skalarni_produkt = smop.matlabarray([])
# ukazi3.m:46
for dfrek in arange(- 1, 1, 0.001).reshape(-1):
    s2 = sin(dot(dot(dot(2, pi), (f2 + dfrek)), i) + faza2 * pi)
    # ukazi3.m:48
    skalarni_produkt[skalarni_produkt.shape[0] + 1] = dot(s1, s2.T)  # Ekvivalentno
# ukazi3.m:49

plt.figure()
plt.plot(arange(- 1, 1, 0.001), skalarni_produkt)
plt.xlabel('frekvencno neujemanje (Hz)')

plt.plot(smop.cat(- 1, 1), smop.cat(0, 0), 'r')
'-------------------------------------------------------------\n% skalarni produkt dveh sinusoid z ENAKIMA FREKVENCAMA IN RAZLIcNIMA FAZAMA\n'
Fvz = 512
# ukazi3.m:55
T = 1
# ukazi3.m:56
i = smop.cat(arange(0.0, T * Fvz)) / Fvz
i.squeeze()
# ukazi3.m:57

f1 = 5
# ukazi3.m:58
A1 = 1
# ukazi3.m:58
f2 = 5
# ukazi3.m:59
faza2 = 0
# ukazi3.m:59
plt.figure()
for faza1 in arange(- 1, 1, 0.05).reshape(-1):
    plt.clf()

    # ustvarimo signal
    x = dot(A1, sin(dot(2 * pi * f1, i) + faza1 * pi))
    x.squeeze()
    # ukazi3.m:64
    s1 = dot(A1, sin(dot(2 * pi * f2, i) + faza2 * pi))
    s1.squeeze()
    # ukazi3.m:66
    Xs1 = dot(s1, x.T)
    # ukazi3.m:68

    plt.subplot(2, 1, 1)

    plt.plot(i.tolist(), x.T.tolist())

    plt.plot(i.tolist(), s1.tolist(), 'r')
    # plt.hold('off')
    plt.xlabel('cas (s)')
    plt.ylabel('Amplituda')
    plt.title(smop.cat('frekvecna = ', str(f1), ', faza = ', str(faza1), ' \\pi'))
    plt.axis('tight')
    # plt.autoscale(False)
    plt.subplot(2, 1, 2)

    plt.autoscale(False)
    plt.plot(smop.cat(0, 0).tolist(), smop.cat(0, Xs1).tolist(), lineWidth=2)
    plt.xlabel('realna os')
    plt.ylabel('imaginarna os')
    # plt.axis(-300, 300, -300, 300)

    plt.xlim(-300, 300)
    plt.ylim(-300, 300)
    # setattr(plt.gca(),'YLim',smop.cat(- 300,300))
    # setattr(plt.gca(),'XLim',smop.cat(- 300,300))
    plt.title(smop.cat('frekvecna sin = ', str(f2), ', faza2 = ', str(faza2), ' \\pi'))
    plt.pause(0.5)

'-------------------------------------------------------------\n% skalarni produkt dveh sinusoid z ENAKIMA FREKVENCAMA IN RAZLIcNIMA FAZAMA\n% analiza s sinusoidami in kosinusoidami\n'
plt.close('all')
Fvz = 512
# ukazi3.m:90
T = 1
# ukazi3.m:91
i = smop.cat(arange(0.0, T * Fvz)) / Fvz
# ukazi3.m:92

f1 = 5
# ukazi3.m:93
A1 = 1
# ukazi3.m:93
f2 = 5
# ukazi3.m:94
faza2 = 0
# ukazi3.m:94
plt.figure()
for faza1 in arange(- 1, 1.05, 0.05).reshape(-1):
    plt.clf()
    # ustvarimo signal
    x = dot(A1, sin(dot(2 * pi * f1, i) + faza1 * pi))
    # ukazi3.m:99
    s1 = dot(1, sin(dot(2 * pi * f2, i) + faza2 * pi))
    # ukazi3.m:101
    c1 = dot(1, cos(dot(2 * pi * f2, i) + faza2 * pi))
    # ukazi3.m:103
    Xs1 = dot(s1, x.T)
    # ukazi3.m:105
    Xc1 = dot(c1, x.T)
    # ukazi3.m:106
    plt.subplot(2, 1, 1)
    plt.plot(i.tolist(), x.T.tolist())

    s1 = s1.squeeze()
    plt.plot(i.tolist(), s1.tolist(), 'r')
    # hold('off')
    plt.xlabel('cas (s)')
    plt.ylabel('Amplituda')
    plt.title(smop.cat('frekvecna = ', str(f1), ', faza = ', str(faza1), ' \\pi'))
    plt.axis('tight')
    plt.subplot(2, 1, 2)
    x = x.squeeze()
    X = fft.fft(x)
    # ukazi3.m:116
    plt.autoscale(False)
    plt.plot(smop.cat(0, Xc1).tolist(), smop.cat(0, Xs1).tolist(), lineWidth=2)
    plt.xlabel('x os')
    plt.ylabel('y os')
    # plt.axis('equal')
    plt.ylim(-300, 300)
    plt.xlim(-300, 300)
    plt.title(smop.cat('frekvecna sin in cos = ', str(f2), ', faza2 = ', str(faza2), ' \\pi'))
    plt.pause(0.5)

'-------------------------------------------------------------\n% skalarni produkt dveh sinusoid z ENAKIMA FREKVENCAMA IN RAZLIcNIMA FAZAMA\n% komplesna analiza s fft.fft\n'
Fvz = 512
# ukazi3.m:128
T = 1
# ukazi3.m:129
i = smop.cat(arange(0.0, T * Fvz)) / Fvz
# ukazi3.m:130

f1 = 5
# ukazi3.m:131
A1 = 1
# ukazi3.m:131
faza1 = 0.5
# ukazi3.m:131
plt.figure()
for faza1 in arange(-1, 1.05, 0.05).reshape(-1):
    plt.clf()
    # ustvarimo sinusiudo
    x = A1 * sin(dot(2 * pi * f1, i) + faza1 * pi)
    # ukazi3.m:136
    plt.subplot(2, 1, 1)
    plt.plot(i.tolist(), x.T.tolist())  # narisemo prvo

    plt.plot(i.tolist(), s1.tolist(), 'r')  # in se drugo sinusoido
    # hold('off')
    plt.xlabel('cas (s)')
    plt.ylabel('Amplituda')
    plt.title(smop.cat('faza = ', str(faza1), ' \\pi'))
    plt.axis('tight')
    plt.subplot(2, 1, 2)
    X = fft.fft(x)

    # ukazi3.m:146

    plt.autoscale(False)
    plt.plot(smop.cat(0, real(X.T[f1])).tolist(), smop.cat(0, imag(X.T[f1])).tolist(), lineWidth=2)
    plt.xlabel('realna os')
    plt.ylabel('imaginarna os')
    # plt.axis('equal')
    plt.ylim(-300, 300)
    plt.xlim(-300, 300)
    plt.pause(0.5)

######################################################
# skalarni produkt dveh sinusoid z ENAKIMA FREKVENCAMA IN RAZLIcNIMA FAZAMA
# komplesna analiza s fft - pogled z druge perspektive (realni in imaginarni del fft-ja)

Fvz = 512
# ukazi3.m:160
T = 1
# ukazi3.m:161
i = smop.cat(arange(0.0, T * Fvz)) / Fvz
# ukazi3.m:162

f1 = 5
# ukazi3.m:163
A1 = 5
# ukazi3.m:163
faza1 = 0.5
# ukazi3.m:163
plt.figure()
for faza1 in arange(- 1, 1, 0.05).reshape(-1):
    # ustvarimo sinusiudo
    x = dot(A1, sin(dot(2 * pi * f1, i) + faza1 * pi))
    # ukazi3.m:168
    X = fft.fft(x)
    # ukazi3.m:170
    plt.subplot(2, 1, 1)
    plt.stem(dot(i / T, Fvz).tolist(), real(X.T).tolist())
    plt.xlabel('Frekvenca (Hz)')
    plt.ylabel('Amplituda')
    plt.title(smop.cat('realni del fft.fft; faza = ', str(faza1), ' \\pi'))
    plt.axis('tight')
    # setattr(plt.gca(),'YLim',smop.cat(- 1500,1500))
    plt.autoscale(False)
    plt.ylim(-1500, 1500)
    plt.subplot(2, 1, 2)
    plt.stem(dot(i / T, Fvz).tolist(), imag(X.T).tolist())
    plt.xlabel('Frekvenca (Hz)')
    plt.ylabel('Amplituda')
    plt.title('imaginarni del fft.fft')
    plt.axis('tight')
    plt.axis('tight')
    # setattr(plt.gca(),'YLim',smop.cat(- 1500,1500))
    plt.autoscale(False)
    plt.ylim(-1500, 1500)
    plt.pause(0.05)

'-------------------------------------------------------------\n% superpozicija sinusoid, cosinusiode in sinusoide\n'
Fvz = 512
# ukazi3.m:187
T = 1
# ukazi3.m:188
i = smop.cat(arange(0.0, T * Fvz)) / Fvz
# ukazi3.m:189

f1 = 5
# ukazi3.m:190
A1 = 5
# ukazi3.m:190
faza1 = 0.5
# ukazi3.m:190
f2 = 32
# ukazi3.m:191
A2 = 3
# ukazi3.m:191
faza2 = 0.3
# ukazi3.m:191
# ustvarimo superpozicijo sinusoid
x = dot(A1, sin(dot(2 * pi * f1, i) + faza1 * pi)) + dot(A2, sin(dot(2 * pi * f2, i) + faza2 * pi))
# ukazi3.m:193
plt.figure()
plt.plot(i.tolist(), x.T.tolist())

plt.xlabel('cas (s)')
plt.ylabel('Amplituda')
plt.axis('tight')
X = fft.fft(x)
# ukazi3.m:201

plt.figure()
plt.subplot(2, 1, 1)
plt.stem(dot(i / T, Fvz).tolist(), real(X.T).tolist())

plt.xlabel('Frekvenca (Hz)')
plt.ylabel('Amplituda')
plt.title('realni del fft.fft')
plt.axis('tight')
plt.subplot(2, 1, 2)
plt.stem(dot(i / T, Fvz).tolist(), - imag(X.T).tolist())

plt.xlabel('Frekvenca (Hz)')
plt.ylabel('Amplituda')
plt.title('imaginarni del fft')
plt.axis('tight')
plt.axis('tight')
################################################################
plt.close('all')
plt.figure()
M = abs(X)
# ukazi3.m:215
# X=X.squeeze()
P = arctan2(imag(X), real(X))
# P=[None]*len(X.tolist()) # Init list of specified size
# for index in arange(0,len(X)-1,1):
#   P[index] = math.atan2(imag(X[index]),real(X[index]))

# ukazi3.m:216

M = M.squeeze()
plt.stem(dot(i / T, Fvz).tolist(), M)

plt.xlabel('Frekvenca')
plt.ylabel('Amplituda')
plt.title('Amplituda fft-ja')
plt.axis('tight')
# niso vse faze zanimive...
plt.figure()
plt.plot(P)

plt.title('Faza fft-ja')
plt.axis('tight')
# le faze tistih sinusoid, ki so dejansko prisotne v signalu
plt.figure()
P1 = multiply(P, (M > 0.1))
# ukazi3.m:230

P1 = P1.squeeze()
plt.stem(dot(i / T, Fvz).tolist(), P1)

plt.xlabel('Frekvenca')
plt.ylabel('Faza')
plt.title('Faza fft-ja')
plt.axis('tight')
'% POJASNILO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
# atan2(imag(X),real(X)) vrne faze za enacbo M*sin(x-pi/2+theta), torej moramo ocenjeni fazi pristeti se pi/2;
#      P = P + pi/2;
# Ne pozabimo tudi, da smo pri generiranju signala x spremenjivko faza2 pomnozili s pi, torej je faza izrazena v enotah pi
'% KONEC POJASNILA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
'-----------------------------------------------------------------------------------------------'
################################################################################################

#             L A S T N O S T I   D F T

################################################################################################
'-----------------------------------------------------------------------------------------------'
# linearnost DFT-ja
x = random.randn(128, 1)
# ukazi3.m:250

X = fft.fft(x)
# ukazi3.m:251

y = random.randn(128, 1)
# ukazi3.m:252

Y = fft.fft(y)
# ukazi3.m:253

a = random.rand(1)
# ukazi3.m:255

b = random.rand(1)
# ukazi3.m:256
z = dot(x, a) + dot(y, b)
# ukazi3.m:258

Z = fft.fft(z)
# ukazi3.m:259

# graficni preizkus linearnosti DFT-ja
plt.figure()
plt.subplot(2, 1, 1)

plt.plot(real(Z), lineWidth=2)
plt.plot(real(dot(X, a) + dot(Y, b)), 'r', lineWidth=1)
plt.xlabel('frekvenca')
plt.ylabel('realni del')
plt.title('fft.fft(a*x + b*y) in a*fft.fft(x) + b*fft.fft(y): primerjava realnih delov')
plt.axis('tight')
plt.subplot(2, 1, 2)

plt.plot(imag(Z), lineWidth=2)
plt.plot(imag(dot(X, a) + dot(Y, b)), 'r', lineWidth=1)
plt.ylabel('imaginarni del')
plt.xlabel('frekvenca')
plt.title('fft.fft(a*x + b*y) in a*fft.fft(x) + b*fft.fft(y): primerjava imaginarnih delov')
plt.axis('tight')
'-----------------------------------------------------------------------------------------------'
# Parsevalov teorem
x = random.randn(128, 1)
# ukazi3.m:280

X = fft.fft(x)
# ukazi3.m:281

energija_v_t_domeni = dot(x.T, x)
# ukazi3.m:282
energija_v_f_domeni = mean(abs(X) ** 2)
# ukazi3.m:283
'-----------------------------------------------------------------------------------------------'
# fft in konvoluciija

smop.clc()

Fs = 44100
# ukazi3.m:290

bits = 16
# ukazi3.m:291

T1 = 3
# ukazi3.m:292
T2 = 2
# ukazi3.m:292

# my_recorder=audiorecorder(Fs,bits,1)
# ukazi3.m:294
disp(smop.cat('Snemam prvi posnetek v dolzini ', str(T1), 's...'))
# recordblocking(my_recorder,T1)
# posnetek1=getaudiodata(my_recorder)
posnetek1 = sd.rec(T1 * Fs, Fs, 1, blocking=True)
# ukazi3.m:298
disp(smop.cat('Snemam drugi posnetek v dolzini ', str(T2), 's...'))
# recordblocking(my_recorder,T2)
# posnetek2=getaudiodata(my_recorder)
posnetek2 = sd.rec(T2 * Fs, Fs, 1, blocking=True)
# ukazi3.m:302
disp('predvajam prvi posnetek')
# my_player=audioplayer(posnetek1,Fs)
myplayer = sd.play(posnetek1, Fs, blocking=True)
# ukazi3.m:305
# playblocking(my_player)
disp('predvajam drugi posnetek')
my_player = sd.play(posnetek2, Fs, blocking=True)
# ukazi3.m:309
# playblocking(my_player)
smop.clc()
disp('konvolucija posnetkov v casovnem prostoru')
efekt1 = smop.matlabarray([])
# ukazi3.m:314
tic = time.time()
efekt1[:, 1] = convolve(posnetek1[:, 1], posnetek2[:, 1])
# ukazi3.m:316
toc = time.time()
t = copy(toc)
# ukazi3.m:317
efekt1 = dot(efekt1 / max(efekt1), max(posnetek1))
# ukazi3.m:318

disp(smop.cat('Potrebovali smo "samo" ', str(round(dot(t, 100)) / 100), ' s'))
disp(smop.cat('... in rezultat se slisi takole:'))
my_player = sd.play(efekt1, Fs, blocking=True)
# ukazi3.m:322
# playblocking(my_player)
disp('konvolucija posnetkov v frekvencnem prostoru')
efekt2 = smop.matlabarray([])
# ukazi3.m:327
tic = time.time()
X = fft.fft(smop.cat([posnetek1[:, 1]], [zeros(len(posnetek2[:, 1]) - 1, 1)]))
# ukazi3.m:329
Y = fft.fft(smop.cat([posnetek2[:, 1]], [zeros(len(posnetek1[:, 1]) - 1, 1)]))
# ukazi3.m:330
efekt2[:, 1] = fft.ifft.fft(multiply(X, Y))
# ukazi3.m:331
toc = time.time()
t = copy(toc)
# ukazi3.m:332
efekt2 = dot(efekt2 / max(efekt2), max(posnetek1))
# ukazi3.m:333

disp(smop.cat('Potrebovali smo ', str(round(dot(t, 100)) / 100), ' s'))
disp(smop.cat('... in rezultat se slisi takole:'))
my_player = sd.play(efekt2, Fs, blocking=True)
# ukazi3.m:337
# playblocking(my_player)
'-----------------------------------------------------------------------------------------------'
# fft vsebina zvocnega posnetka

smop.clc()

Fs = 44100
# ukazi3.m:346

bits = 16
# ukazi3.m:347

T1 = 5
# ukazi3.m:348
# my_recorder=audiorecorder(Fs,bits,1)
# ukazi3.m:350
disp(smop.cat('Snemam prvi posnetek v dolzini ', str(T1), 's...'))
# recordblocking(my_recorder,T1)
posnetek = sd.rec(T1 * Fs, Fs, 1)
# ukazi3.m:354
plt.figure()
plt.plot(posnetek)
X = (fft.fft(posnetek))
# ukazi3.m:359

N = size(posnetek, 1)
# ukazi3.m:361
dF = Fs / N
# ukazi3.m:362

f = smop.matlabarray(smop.cat(arange(dF, Fs / 2, dF), arange(Fs / 2 - dF, 0, - dF)))
# ukazi3.m:363
plt.figure()
plt.plot(f, abs(X) / N)
plt.xlabel('Frekvenca (H)')
plt.title('Odziv magnitude')
