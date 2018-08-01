%----------------------------------------------------------------------------------------
% Vzor�enje in Nyquistov teorem;
Fvz = 100; % Frekvenca vzor�enja (v Hz)
T = 1; % dol�ina signala (v s)
i = [0:1:T*Fvz-1]/Fvz; % vektor �asovnih indeksov
f1=5; % frekvenca sinusoide
A1=1; % amplituda sinusoide
faza1=0.0; % faza sinusoide

% izris sinuside pri razlicnih fazah......................................................
figure
for faza1=0:0.1:6; 
    s=A1*sin(2*pi*f1*i+faza1*pi); 
    plot(i,s); 
    axis tight; 
    set(gca,'YLim',[-1,1]);
    title(['Fvz = ' num2str(Fvz) ' Hz, Frekvenca = ' num2str(f1) ' Hz, faza = ' num2str(faza1) '\pi']);
    xlabel('�as (s)');
    ylabel('amplituda (dB)');
    pause; 
end

% izris sinusid pri razlicnih frekvencah...............................................
faza1 = 0;
figure
for f1=0:1:Fvz; 
    s=A1*sin(2*pi*f1*i+faza1*pi); 
    plot(i,s); 
    axis tight; 
    set(gca,'YLim',[-1,1]);
    title(['Fvz = ' num2str(Fvz) ' Hz, Frekvenca = ' num2str(f1) ' Hz, faza = ' num2str(faza1) '\pi']);
    xlabel('�as (s)');
    ylabel('amplituda (dB)');
    pause(0.05); 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%% izris sinusoid s frekvenco f1 in Fvz-f1.........................
figure; hold on; %
f1=1; 
s1=sin(2*pi*f1*i+faza1*pi); 
plot(i,s1,'b'); 
f2=Fvz-f1; 
s2=sin(2*pi*f2*i+faza1*pi); 
plot(i,s2,'r'); 
xlabel('�as (s)'); ylabel('Amplituda');
title(['Fvz = ' num2str(Fvz) ' Hz, Frekvenca1 = ' num2str(f1) ' Hz,  Frekvenca2 = ' num2str(f2) ' Hz, faza = ' num2str(faza1) '\pi']);
axis tight;

%----------------------------------------------------------------------------------------

% glej tudi primera v Mathematici: (WheelIllusion.nbp in SamplingTheorem.nbp)


% V Z O R � E N J E    Z V O K A
%-----------------------------------------------------------------------------------------
% vzor�enje zvoka 
Fs = 44100; % vzor�evalna frekvenca
bits = 16; % bitna lo�ljivost
nchans = 1; % 1 (mono), 2 (stereo).
recorder = audiorecorder(Fs,bits,nchans);
recordblocking(recorder,5);
posnetek = getaudiodata(recorder);
figure; plot(posnetek);

sound(1*posnetek,44100);

sound(1*posnetek,44100/2);

sound(1*posnetek,2*44100);


%-----------------------------------------------------------------------------------------
% ali zaznate fazne spremembe? Spreminjajte faza1 med 0 in 2.0 in po�enite ta demo...
Fvz = 44100;% vzor�evalna frekvenca
T = 3; % �as v sekundah
i = [0:T*Fvz-1]/Fvz; % vektor �asovnih indeksov
f1 = 500; % frekvenca sinusoide
A1 = 0.3; % amplituda sinusoide
faza1 = 1.0; % faza sinusoide

s=A1*sin(2*pi*f1*i+faza1*pi); % tvrojenje sinusoide
s2=A1*sin(2*pi*f1*i+0*pi); % tvrojenje sinusoide

sound([s s2],Fvz);


%-----------------------------------------------------------------------------------------
% trije poskusi: 1. f1 = 50;
%                2. f1 = 450;
%                3. f1 = 1450;
%                4. f1 = 2450;   
Fvz = 44100;% vzor�evalna frekvenca
T = 3;
i = [0:T*Fvz-1]/Fvz; % vektor �asovnih indeksov
f1=50; % frekvenca sinusoide
A1=5.5; % amplituda sinusoide
faza1=0.0; % faza sinusoide
f2=f1+1; % frekvenca druge sinusoide

s1=A1*sin(2*pi*f1*i+faza1*pi); % tvrojenje prve sinusoide
s2=A1*sin(2*pi*f2*i+faza1*pi); % tvrojenje druge sinusoide
sound([s1 s2],Fvz);


%-----------------------------------------------------------------------------------------
% ali zaznate zvok netopirja pri 90000 Hz? Nyquist?
Fvz = 44100;% vzor�evalna frekvenca
T = 3;
i = [0:T*Fvz-1]/Fvz; % vektor �asovnih indeksov
fnetopir = 140000; % frekvenca sinusoide
A1=5.5; % amplituda sinusoide
faza1=1.0; % faza sinusoide

s=A1*sin(2*pi*fnetopir*i+faza1*pi); % tvorjenje sinusoide
sound(s,Fvz);


% izris sinuside pri razlicnih fazah (verzija 2)......................................................
Fvz = 100;
T = 1;
i = [0:T*Fvz-1]/Fvz; % vektor �asovnih indeksov
f1=5; % frekvenca sinusoide
A1=5; % amplituda sinusoide
faza1=0.0; % faza sinusoide


% spremninjanje frekvence...
close all;
figure;
for f1=0:Fvz; 
    s=sin(2*pi*f1*i+faza1*pi); 
    
    % izris v �asovni domeni
    subplot(2,1,1);
    plot(s); 
    axis tight; 
    set(gca,'YLim',[-1,1]);
    title(['�asovna domena: Fvz = ' num2str(Fvz) ' Hz, Frekvenca = ' num2str(f1) ' Hz, faza = ' num2str(faza1) '\pi']);
    
    % izris v frekven�ni domeni
    subplot(2,1,2);
    plot(abs(fft(s)),'r'); 
    axis tight; 
    set(gca,'YLim',[-1,1]);
    title(['Frekven�na domena (abs): Fvz = ' num2str(Fvz) ' Hz, Frekvenca = ' num2str(f1) ' Hz, faza = ' num2str(faza1) '\pi']);
    
    pause; 
end


% in faze... (ve� o tem na naslednjih vajah)
f1=5; % frekvenca sinusoide
A1=5; % amplituda sinusoide
faza1=0.0; % faza sinusoide
close all;
figure;
for faza1=0:0.1:2; 
    s=sin(2*pi*f1*i+faza1*pi); 
    
    % izris v �asovni domeni
    subplot(2,1,1);
    plot(s); 
    axis tight; 
    set(gca,'YLim',[-1,1]);
    title(['�asovna domena: Fvz = ' num2str(Fvz) ' Hz, Frekvenca = ' num2str(f1) ' Hz, faza = ' num2str(faza1) '\pi']);
    
    % izris v frekven�ni domeni
    subplot(2,1,2);
    plot(abs(fft(s)),'r'); 
    axis tight; 
    set(gca,'YLim',[-1,1]);
    title(['Frekven�na domena (abs): Fvz = ' num2str(Fvz) ' Hz, Frekvenca = ' num2str(f1) ' Hz, faza = ' num2str(faza1) '\pi']);
    
    pause; 
end



% S L I K E
%-----------------------------------------------------------------------------------------
% vzor�enje slik in Moire 
addpath('D:\Dropbox\FERI\mag\ROSIS\1-Vzorcenje')
A = imread('Moire.jpg');
figure; 
imshow(A);
title('originalna slika'); 


pvz = 3; % faktor podvzor�enja
figure;
imshow(A(1:pvz:end,1:pvz:end,:));
title(['podvzor�ena slika: faktor podvzor�enja ' num2str(pvz)]); 
 
st_bit = 0; % nova bitna lo�ljivost 
kvant = 2^(9-st_bit);
figure;
imshow((A(:,:,:)/kvant)*kvant);
title(['slika pri bitni lo�ljivosti ' num2str(st_bit)]); 

figure; subplot(2,2,1);
imshow((A(:,:,:)/kvant)*kvant);
title(['slika pri bitni lo�ljivosti ' num2str(st_bit)]); 
subplot(2,2,2);
imshow((A(:,:,1)/kvant)*kvant);
title(['ravnina R pri bitni lo�ljivosti ' num2str(st_bit)]); 
subplot(2,2,3);
imshow((A(:,:,2)/kvant)*kvant);
title(['ravnina R pri bitni lo�ljivosti ' num2str(st_bit)]); 
subplot(2,2,4);
imshow((A(:,:,3)/kvant)*kvant);
title(['ravnina R pri bitni lo�ljivosti ' num2str(st_bit)]); 


%-----------------------------------------------------------------------------------------
% spekter slik, Moire in Diskretna Fourierova transformacija (fft2) 

addpath('D:\Dropbox\FERI\mag\ROSIS\1-Vzorcenje')
A = imread('Moire.jpg');
figure; 
imshow(A);
title('originalna slika'); 

close all;
B = double(A(:,:,1));
figure; mesh(abs(fft2( B - mean(B(:)),100,100 ))); title('R ravnina')
B = double(A(:,:,2));
figure; mesh(abs(fft2( B - mean(B(:)),100,100 ))); title('G ravnina')
B = double(A(:,:,3));
figure; mesh(abs(fft2( B - mean(B(:)),100,100 ))); title('B ravnina')

% prevzor�ena slika..................................
pvz = 4;
B = double(A(1:pvz:end,1:pvz:end,1));
figure; mesh(abs(fft2( B - mean(B(:)),100,100 ))); title(['R ravnina, po podvzor�enju s faktorjem ' num2str(pvz)]);
B = double(A(1:pvz:end,1:pvz:end,2));
figure; mesh(abs(fft2( B - mean(B(:)),100,100 ))); title(['G ravnina, po podvzor�enju s faktorjem ' num2str(pvz)]);
B = double(A(1:pvz:end,1:pvz:end,3));
figure; mesh(abs(fft2( B - mean(B(:)),100,100 ))); title(['B ravnina, po podvzor�enju s faktorjem ' num2str(pvz)]);

%-----------------------------------------------------------------------------------------
% podvzor�enje slik in operator povpre�enja 
addpath('D:\Dropbox\FERI\mag\ROSIS\1-Vzorcenje')
A = imread('Moire.jpg');
figure; 
imshow(A);
title('originalna slika'); 

pvz = 3; % faktor podvzor�enja
figure;
imshow(A(1:pvz:end,1:pvz:end,:));
title(['podvzor�ena slika: faktor podvzor�enja ' num2str(pvz)]); 
 
% operator povpre�enja (verzija 1)
D=3; % premer lokalne okolice piksla, na kateri se izra�una povpre�na verdnost
B=[];
for r=1:size(A,1)-D+1
    for c=1:size(A,2)-D+1
        C=A(r+0:D-1,c+[0:D-1],1); % R ravnina
        B(r,c,1) = mean(C(:));
        C=A(r+[0:D-1],c+[0:D-1],2); % G ravnina
        B(r,c,2) = mean(C(:));
        C=A(r+[0:D-1],c+[0:D-1],3); % B ravnina
        B(r,c,3) = mean(C(:));
    end
end
figure;
imshow(uint8(B));
title('zglajena slika'); 

% operator povpre�enja (verzija 2)
% isti operator povpre�enja kot zgoraj, implementiran nekoliko druga�e (veliko hitrej�a izvedba)
D=3; % premer lokalne okolice piksla, na kateri se izra�una povpre�na vrednost
B=[];
B(:,:,1) = conv2(double(A(:,:,1)),ones(D,D)/D^2);
B(:,:,2) = conv2(double(A(:,:,2)),ones(D,D)/D^2);
B(:,:,3) = conv2(double(A(:,:,3)),ones(D,D)/D^2);
B = uint8(B);
figure;
imshow(B);
title('zglajena slika'); 

% prikaz
pvz = 3; % faktor podvzor�enja
figure; subplot(1,2,1)
imshow(A(1:pvz:end,1:pvz:end,:));
title(['podvzor�ena slika: faktor podvzor�enja ' num2str(pvz)]); 
subplot(1,2,2)
imshow(B(1:pvz:end,1:pvz:end,:));
title(['zglajena podvzor�ena slika: faktor podvzor�enja ' num2str(pvz)]); 