function g = music_lp(filename)

    [y, Fs] = audioread(filename);
    info = audioinfo(filename);
    
    t = 0:seconds(1/Fs):seconds(info.Duration);
    t = t(1:end-1);
    plot(t,y)
    
    [numSample, numChannel] = size(y);
    numFreq = (numSample-2)/2 + 1;

end