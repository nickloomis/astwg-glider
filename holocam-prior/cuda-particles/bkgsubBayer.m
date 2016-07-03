function S = bkgsubBayer(H,bkg,m)
%
% Usage: S = bkgsubBayer(H, bkg, bkgmeans);
%
% Subtracts a background, bkg, from an image (hologram), H, by scaling each
% of the color channels. If the optional bkgmeans are provided (the mean
% values of bkg channels, as returned by bayerHist3FR), they can be used to
% speed up the computation slightly.
%

% Nick Loomis, Summer 2010
%

mh = zeros(1,4);
mh(1) = mean2(H(1:2:end,1:2:end));
mh(2) = mean2(H(1:2:end,2:2:end));
mh(3) = mean2(H(2:2:end,1:2:end));
mh(4) = mean2(H(2:2:end,2:2:end));

if ~exist('m', 'var')
    m = zeros(1,4);
    m(1) = mean2(H(1:2:end,1:2:end));
    m(2) = mean2(H(1:2:end,2:2:end));
    m(3) = mean2(H(2:2:end,1:2:end));
    m(4) = mean2(H(2:2:end,2:2:end));
end

%make sure we're in single precision, unless bkg or H is double (and thus
%forces us back to double)
mh = single(mh);
m = single(m);

S = zeros(size(H),'single');
S(1:2:end,1:2:end) = H(1:2:end,1:2:end) - bkg(1:2:end,1:2:end)*mh(1)/m(1);
S(1:2:end,2:2:end) = H(1:2:end,2:2:end) - bkg(1:2:end,2:2:end)*mh(2)/m(2);
S(2:2:end,1:2:end) = H(2:2:end,1:2:end) - bkg(2:2:end,1:2:end)*mh(3)/m(3);
S(2:2:end,2:2:end) = H(2:2:end,2:2:end) - bkg(2:2:end,2:2:end)*mh(4)/m(4);

