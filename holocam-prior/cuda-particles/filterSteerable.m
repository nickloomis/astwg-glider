function [s,so,sx,sy] = filterSteerable(img,sigma)

if nargin<2
    sigma = 1.5;
end

sfilt = steerableDeriv([],sigma);
sx = imfilter(img,sfilt,'same','replicate');
sy = imfilter(img,sfilt','same','replicate');
s = sqrt(sx.^2 + sy.^2);
so = atan2(sy,sx);