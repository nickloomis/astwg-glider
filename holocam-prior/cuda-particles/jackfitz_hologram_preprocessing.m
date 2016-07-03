%hologram processing script

datadir = 'C:\Documents and Settings\Administrator\My Documents\Hologram data\2010-06-13 deep cast, station 2 (dive 3, 270, 2km)';
starttime = [2010 06 13 9 50 47];
delaytime = [0 0 0 0 6 8.7];
timezone = -5; %to convert to GMT/UZT

firstgoodholo = 1007480; %number on the 3FR file
lastgoodholo = 1008330;


nbkg = 30; %number of holograms to use in estimating a stationary background

%rip previews of the files
rip3FRpreviews(datadir);

%read, adjust the times
[adjtime, dir3fr] = read3FRtimes(datadir, starttime, timezone, delaytime);


%create a random sampling of holograms to use to estimate the background
deltagood = lastgoodholo - firstgoodholo + 1; %number of holograms
pholo = randperm(floor(deltagood*0.8)) + ceil(deltagood*.1);
%random permutation over the middle 80% of the good data: offset from
%the first good holo

numofdir = zeros(numel(dir3fr,1));
for j=1:numel(dir3fr)
    filename = dir3fr(j).name;
    numofdir(j) = str2num( filename(2:end-4));
end

wbp = waitbar(0,'Creating background...');
%initialize the bkg
idx = find(pholo(1) + firstgoodholo == numofdir);
bkg = readDNG([datadir, filesep, dir3fr(idx).name]);

waitbar(1/nbkg,wbp);
for j=2:nbkg
    idx = find(pholo(j) + firstgoodholo == numofdir);
    bkg = bkg + readDNG([datadir, filesep, dir3fr(idx).name]);
    waitbar(j/nbkg,wbp);
end
delete(wbp);
bkg = bkg/nbkg;

[histbkg, histbins, meanbkg] = bayerHist3FR(bkg);

save([datadir, filesep, 'background.mat'], 'bkg', 'nbkg', ...
    'histbkg', 'histbins', 'meanbkg');
bsm = imresize(bkg,.1);
imwrite(flipud(normmat(rot90(double(bsm),-1))),...
    [datadir, filesep, 'background_preview.bmp'],'bmp');

