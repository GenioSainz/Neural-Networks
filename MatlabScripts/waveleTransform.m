

clear all;clc;

N        = 500;
fs       = 100;
timeS    = linspace(0,2,N);
F        = [4,8,12,16];
A        = F;
DC_value = 0;
y        = getComposeSignal(F,A,DC_value,timeS);

figure(1);clf
hold on;box on;grid on

plot(timeS,y)

num_frex = 500;

min_freq = 2;
max_freq = 20;

min_cycle = 4;
max_cycle = 10;

frex      = linspace(min_freq,max_freq,num_frex);
nCycles   = linspace(min_cycle,max_cycle,num_frex);
timeW     = -2:1/fs:2;
half_wave = (length(timeW)-1)/2;

nKern = length(timeW);
nData = N;
nConv = nData+nKern-1;

Y = fft(y,nConv);

DWT   = zeros(num_frex,nData);

for fi=1:length(frex)
    
    % create wavelet and get its FFT
    
    wavelet = wavelet_fun(frex(fi),nCycles(fi),timeW);
    W       = fft(wavelet,nConv);
    
    % run convolution
    as = ifft(W.*Y,nConv);
    as = as(half_wave+1:end-half_wave);
    %as = reshape(as,EEG.pnts,EEG.trials);
    
    % put power data into big matrix
    DWT(fi,:) = mean(abs(as).^2,2);
end


figure(2);clf
imagesc(DWT)
xlabel('Time (ms)'),
ylabel('Frequency (Hz)')


function wavelet = wavelet_fun(frec,nCyles,time)

    s     = nCyles/(2*pi*frec);
    imag  = exp(2*pi*1j*frec*time);
    gauss = exp((-time.^2)./(2*s^2));

    wavelet = gauss.*imag;

end


function y = getComposeSignal(F,A,DC_value,t)

    N = length(t);
    y = zeros(1,N) + DC_value;

    for k=1:length(A)

        y = y + A(k)*sin(2*pi*F(k)*t);
    end
end