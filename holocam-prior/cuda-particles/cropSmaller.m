function C = cropSmaller(img, starget)
%
% Usage: C = cropSmaller(img, starget)
%
% Crops an image, img, to a smaller (square) target size, starget. A Bayer
% pattern is assumed for the image, so that the crop is shifted to have the
% same Bayer layout as the original raw image.
%

% Nick Loomis, Summer 2010
%

simg = size(img);
pixfromleft = floor( (simg(2) - starget) / 4) * 2;
pixfromtop = floor( (simg(1) - starget) / 4) *2;

C = img(pixfromtop : pixfromtop + starget - 1, ...
    pixfromleft : pixfromleft + starget - 1);
