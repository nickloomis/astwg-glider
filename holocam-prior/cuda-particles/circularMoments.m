function C = circularMoments(img, xc, yc, r)

s = size(img);
x = 1:s(2);
y = 1:s(1);
[X,Y] = meshgrid(x,y);
%R = sqrt((X-xc).^2 + (Y-yc).^2);
%T = atan2(Y-yc,X-xc);
%mask = R<=r; %mask
mask = circimg([s(2),s(1)], r, xc-s(2)/2, yc-s(1)/2);

X = X(:); Y = Y(:);
%R = R(:); T = T(:);
M = mask(:); %radius weights
I = img(:); %grayscale weights
W = M.*I; %the net grayscale weights using the mask

%raw moments
M00 = sum(W);
M01 = sum(Y.*W);
M10 = sum(X.*W);
M11 = sum(X.*Y.*W);
M20 = sum(X.^2.*W);
M02 = sum(Y.^2.*W);

%centroids
xbar = M10/M00;
ybar = M01/M00;

%scaled central moments
mu20prime = M20/M00 - xbar^2;
mu02prime = M02/M00 - ybar^2;
mu11prime = M11/M00 - xbar*ybar;

C = [mu20prime, mu11prime; mu11prime, mu02prime];


%rotational moments
% D00 = sum(M.*I); %mass
% D20 = sum(R.^2.*M.*I);
% D22 = sum(R.^2.*exp(1i*T*2).*M.*I);
% D31 = sum(R.^3.*exp(1i*T).*M.*I);
% D33 = sum(R.^3.*exp(1i*T*3).*M.*I);
% D40 = sum(R.^4.*M.*I);
% D42 = sum(R.^4.*exp(1i*T*2).*M.*I);
% D44 = sum(R.^4.*exp(1i*T*4).*M.*I);
% D = abs([D20, D22, D31, D33, D40, D42, D44])./[D00^2,D00^2,D00^3,D00^3,D00^4,D00^4,D00^4];
%normalized to the D00 magnitude...
%note that D20, D40 are rotationally symmetric; D22, D33, D42, and D44
%should show indications of antisymmetric or mirror symmetric behavior.


%moments in the x, y directions for pixels
%ix = (X-xc).*I;
%iy = (Y-yc).*I;
%x1 = sum(M.*ix)/sum(M);
%y1 = sum(M.*iy)/sum(Y);
%x2 = sqrt(sum(M.*(ix - x1).^2)/sum(M));
%y2 = sqrt(sum(M.*(iy - y1).^2)/sum(M));
%C = [x2^2, x1*y1; x1*y1, y2^2];