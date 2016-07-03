%everything that needs to happen to process a data directory

%datadir = 'C:\Documents and Settings\Administrator\My Documents\Hologram data\2010-06-14 deep cast, morning (dive 5, 1km)';
%datadir = 'C:\Documents and Settings\Administrator\My Documents\Hologram data\2010-06-18 shallow dive with all instruments (dive 9)';
%datadir = 'C:\Documents and Settings\Administrator\My Documents\Hologram data\2010-06-19 deep dive, morning (dive 10)';
%datadir = 'C:\Documents and Settings\Administrator\My Documents\Hologram data\2010-06-16 deep cast (dive 6, 2km)';
%datadir = 'C:\Documents and Settings\Administrator\My Documents\Hologram data\2010-06-17 deep dive, morning (dive 7)';
%datadir = 'C:\Documents and Settings\Administrator\My Documents\Hologram data\2010-06-17 shallow dive on ctd, evening (dive 8)';
%datadir = 'C:\Documents and Settings\Administrator\My Documents\Hologram data\2010-06-20 deep cast, evening (dive 13)';
%datadir = 'C:\Documents and Settings\Administrator\My Documents\Hologram data\2010-06-19 deep dive, afternoon  (dive 11)';
datadir = 'C:\Documents and Settings\Administrator\My Documents\Hologram data\2010-06-20 deep cast, morning (dive 12)';



%get the rest of the data (specific to the dive directory)
jack_fitz_readHoloConfigScript;

% holostarttime = [2010 06 19 14 42 30];
% holotimezone = -5; %CDT to GMT conversion
% delaytime = [0 0 0 0 12 7]; 
% ctdstarttime = [2010 06 19 14 43 58];
% ctdtimezone = -5; 
% 
% ctdfilename = 'Site-5nm-Cast-1-061910_144358.txt';
% divenum = 11;
% 
% housingsepinches = 11.5; %distance between the front faces of the housings

%nbkg = 30; %maximum number of images to use in creating backgrounds
%npix = 2048; %number of pixels used in the reconstruction steps
%localdist = 150; %number of pixels to be considered "local" for local-area statistics
bilateralSigma = 50;
bilateralRangeSteps = 10;
bilateralnpix = 1024; %based on available matlab memory
%edgeZScore = 3;
blobimgmargin = 10;
pixelsize = 6.8; %size in microns
roverest = 1.2; %amount that the random algorithm tends to overestimate radius
roverestp = 1.5; %amount that the phase-coded detection tends to overestimate radius

% end of configuration section %


         
         
%get the CTD data
ctdalldata = importdata([datadir, filesep, ctdfilename], ' ', 1);
ctddatasize = size(ctdalldata.data);
ctdtimefromstart = ctdalldata.data(:,1);
ctddepthft = ctdalldata.data(:,2);
ctdtemp = ctdalldata.data(:,3);
ctdcond = ctdalldata.data(:,4);
ctdsal = ctdalldata.data(:,5);
ctdturb = ctdalldata.data(:,ctdturbcol); %the 2nd to last column
ctddens = ctdalldata.data(:,ctddenscol); %the next to last colmn


%plots of some basic CTD data
figure(1); subplot(1,3,1);
plot(ctdtemp,ctddepthft,'r','linewidth',2);
xlabel('Temperature (C)'); ylabel('Depth (ft)'); axis ij;
title('Temperature'); axis([0,max(ctdtemp),0,max(ctddepthft)]);
subplot(1,3,2); plot(ctdsal,ctddepthft,'g','linewidth',2);
xlabel('Salinity (psu)'); ylabel('Depth (ft)'); axis ij;
title('Salinity'); axis([30,40,0,max(ctddepthft)]);
subplot(1,3,3); plot(ctddens,ctddepthft,'b','linewidth',2);
xlabel('Density (km/m^3)'); ylabel('Depth (ft)'); axis ij;
title('Raw density'); axis([1020,1040,0,max(ctddepthft)]);
set(gcf,'Position',[150,150,900,400]);
saveplot([datadir,filesep,'ctd_TSD']);

clf();
plot(ctdsal,ctdtemp,'b.','linewidth',2);
axis([30,40,0,max(ctdtemp)]);
ylabel('Temperature (C)'); xlabel('Salinity (psu)'); title('TS diagram');
set(gcf,'Position',[150,150,500,500]);
saveplot([datadir,filesep,'ctd_tsdiagram']);


%connect up the holocam and ctd information
[adjtime, dir3fr, hdatenum] = read3FRtimes(datadir, holostarttime, holotimezone, delaytime);
nholos = numel(dir3fr);
ctddatenum = datenum(ctdstarttime) - ctdtimezone/24 + ctdtimefromstart/(24*60*60);
holodepth = interp1(ctddatenum,ctddepthft,hdatenum);
holotemp = interp1(ctddatenum,ctdtemp,hdatenum);
holosal = interp1(ctddatenum,ctdsal,hdatenum);
holodens = interp1(ctddatenum,ctddens,hdatenum);
holoturb = interp1(ctddatenum,ctdturb,hdatenum);


%plot the profile for where we got holograms
clf();
plot(ctddatenum,ctddepthft,'r','linewidth',2); hold on;
plot(hdatenum,holodepth,'kx'); hold off;
xlabel('Sample time (GMT)','fontsize',12);
ylabel('depth (ft)','fontsize',12); axis ij;
title(['Dive ',num2str(divenum),' holography profile'],'fontsize',14);
datetick('x', 15); grid on; 
axis([min(min(hdatenum),min(ctddatenum)), max(max(hdatenum),max(ctddatenum)),0,max(ctddepthft)*1.05]);
legend('CTD depth','Hologram capture','Location','Southwest');
set(gcf,'Position',[150,150,900,500]);
saveplot([datadir,filesep,'holography_captureDepth_profile']);


%rip previews of the holograms for other people to use
previewdir = [datadir,filesep,'previews'];
[~,~,~] = mkdir(previewdir);
rip3FRpreviews(datadir,previewdir);


%decide which holograms are the 'best' to look at
%1) needs to have been captured when we were in the water column
%2) background type: consistent lighting, 2-up, overblown, other
datastartdatenum = max(min(hdatenum), min(ctddatenum));
dataenddatenum = min(max(hdatenum), max(ctddatenum));
holovalid = (hdatenum>=datastartdatenum).*(hdatenum<=dataenddatenum);
holobkgtype = zeros(nholos,1);
wbh = waitbar(0,'Determining background type...');
for j=1:nholos
    if holovalid(j)
        previmg = imread([datadir,filesep,dir3fr(j).name],'tiff');
        holobkgtype(j) = backgroundType3FR(previmg(:,:,1));
    end
    waitbar(j/nholos,wbh);
end
delete(wbh);


%save some information about the holograms themselves
holonamenum = zeros(nholos,1);
for j=1:nholos
    holonamenum(j) = str2num(dir3fr(j).name(2:end-4));
end
holodatahdr = {'Hologram','Data number','Serial Date',...
    'GMT Hour','GMT Min','GMT Sec','Depth (ft)','Temperature (C)','Salinity (psu)',...
    'Raw Density (km/m3)','Turbidity (NTU)','Background type'};
holodatamatrix = [holonamenum, [1:nholos]', hdatenum, adjtime(:,4),...
    adjtime(:,5), adjtime(:,6), holodepth, holotemp, holosal, ...
    holodens, holoturb, holobkgtype];
writeCSV([datadir, filesep, 'hologram_sampleData.csv'], holodatahdr, holodatamatrix);
    

%average together some backgrounds
diveidx = find((holobkgtype==1) + (holobkgtype==2));
bkg1idx = find(holobkgtype==1);
bkg2idx = find(holobkgtype==2);
nbkg1 = min(nbkg,numel(bkg1idx));
nbkg2 = min(nbkg,numel(bkg2idx));
[~,holonpix] = readDNGInfo([datadir,filesep,dir3fr(1).name]);

wbh = waitbar(0,'Estimating normalized bkg...');

if ~exist([datadir,filesep,'bkgnorm.mat'],'file');
    bkgnorm = zeros(holonpix,'single');
    bkgidx = randperm(numel(diveidx));
    
    for j=1:nbkg
        H = readDNG([datadir,filesep,dir3fr(diveidx(bkgidx(j))).name]);
        bkgnorm = bkgnorm + normalizeBayerRow(H,20);
        waitbar(j/nbkg,wbh);
    end
    bkgnorm = bkgnorm/nbkg;
    waitbar(1,wbh,'Saving normalized background');
    save([datadir,filesep,'bkgnorm.mat'],'bkgnorm');
    imwrite(rot90(normmat(imresize(bkgnorm,.125)),1),[datadir,filesep,'background_norm_preview.png'],'png');
    clear bkgnorm;
end

% 
% waitbar(0,wbh,'Estimating Background type 1...');
% 
% if ~exist([datadir,filesep,'background_type1.mat'],'file');
%     bkg1 = zeros(holonpix,'single');
%     bkgidx = randperm(nbkg1);
% 
%     for j=1:nbkg1
%         H = readDNG([datadir,filesep,dir3fr(bkg1idx(bkgidx(j))).name]);
%         bkg1 = bkg1 + H/mean2(H); %normalize and add
%         waitbar(j/nbkg1,wbh);
%     end
%     bkg1 = bkg1/nbkg1;
%     waitbar(1,wbh,'Saving Background type 1');
%     save([datadir,filesep,'background_type1.mat'],'bkg1');
%     imwrite(rot90(normmat(imresize(bkg1,.125)),1),[datadir,filesep,'background_type1_preview.png'],'png');
%     clear bkg1;
% end
% 
% waitbar(0,wbh,'Estimating Background type 2...');
% 
% if ~exist([datadir,filesep,'background_type2.mat'],'file')
%     bkg2 = zeros(holonpix,'single');
%     bkgidx = randperm(nbkg2);
%     waitbar(0,wbh);
%     for j=1:nbkg2
%         H = readDNG([datadir,filesep,dir3fr(bkg2idx(bkgidx(j))).name]);
%         bkg2 = bkg2 + H/mean2(H);
%         waitbar(j/nbkg2,wbh);
%     end
%     bkg2 = bkg2/nbkg2;
%     waitbar(1,wbh,'Saving Background type 2');
%     save([datadir,filesep,'background_type2.mat'],'bkg2');
%     imwrite(rot90(normmat(imresize(bkg2,.125)),1),[datadir,filesep,'background_type2_preview.png'],'png');
%     clear bkg2;
% end

delete(wbh);





%work through holograms looking for oil droplets

%calculate depths to check
zmin = 40; %in mm
zmax = 40 + housingsepinches*25.4/1.33;
zsphhat = 387; %in mm
zobj = zmin:.2:zmax; %object distances
zpl = 1./(1./zobj - 1/zsphhat);


%write out the metadata information
jack_fitz_writeMetadataFile;




%create a directory for results
[~,~,~] = mkdir([datadir,filesep,'results']); %quiet any "already exists" 
%messages by writing the error message to a NULL instead of the stdout...?


%eliminate holograms from the list of to-dos if they've been processed
dirresults = dir([datadir,filesep,'results',filesep,'diameters_*.csv']);
donenums = zeros(numel(dirresults),1);
for j=1:numel(dirresults)
    donenums(j) = str2num(dirresults(j).name(11:end-4));
end
doneidx = zeros(numel(donenums),1);
for j=1:numel(donenums)
    doneidx(j) = find(holonamenum == donenums(j));
end
for j=1:numel(doneidx)
    idx = find(bkg1idx == doneidx(j));
    if ~isempty(idx)
        bkg1idx(idx) = [];
    end
    idx = find(bkg2idx == doneidx(j));
    if ~isempty(idx)
        bkg2idx(idx) = [];
    end
end


%start with bkg type1 images, they're "nicer" (can use them directly with
%the holograms)
powOrder_type1 = 4;
powOrder_type2 = 2;
%load([datadir,filesep,'background_type1.mat']);
%S = cropSmaller(bkg1,npix);
%clear bkg1;

load([datadir,filesep,'bkgnorm.mat']);
S = cropSmaller(bkgnorm,npix);
S = S/mean2(S);
clear bkgnorm;


tictocs1 = zeros(numel(bkg1idx),1);
wbh = waitbar(0,'Finding particles...');
set(wbh,'position',[660,670,270,56]);
 
%for pnum=1:numel(bkg1idx)
for pnum = 1:numel(diveidx)
    
    %grab the start time
    tic; 
    
    %read in the hologram
    %H = readDNG([datadir,filesep,dir3fr(bkg1idx(pnum)).name]);
    H = readDNG([datadir,filesep,dir3fr(diveidx(pnum)).name]);
    N = normalizeBayerRow(H);
    C = cropSmaller(N,npix);
    clear N H;

    
    %normalize the intensity
%    C = C/mean2(H) - S;
%    clear H
    C = C - S;
        
    %find in-focus edges
    if holobkgtype(diveidx(pnum))==1
        powOrder = powOrder_type1;
    else
        powOrder = powOrder_type2;
    end
    [simcuda, sidxcuda, rmincuda, ridxcuda] = maxSIMfm(5 + C, zpl/1000, 658e-9, 6.8e-6, powOrder, .001, 2, 500);
    
    
    %smooth out the noise
    [Msim,Ssim] = localstatsfilt_cuda(simcuda,localdist);
    simcudanorm = simcuda./Msim; 
    Ssim = Ssim./Msim;
    edgemin = double(min(simcudanorm(:)));
    edgemax = double(max(simcudanorm(:)));
%    bsim = bilateralFilter(double(simcudanorm),double(simcudanorm),edgemin, edgemax, bilateralSigma, (edgemax - edgemin)/bilateralRangeSteps);
%   dang, keep having memory issues with interpn. try sectioning off the
%   bilateral. (if this doesn't work, it's mex time!)
    bsim = single( blockedBilateral(simcudanorm, edgemin, edgemax, bilateralSigma, (edgemax - edgemin)/bilateralRangeSteps, bilateralnpix) );
%     
%     bsim = zeros([npix,npix],'single');
%     bsim(1:npix/2,1:npix/2) = single( bilateralFilter(double(simcudanorm(1:npix/2,1:npix/2)),double(simcudanorm(1:npix/2,1:npix/2)),edgemin, edgemax, bilateralSigma, (edgemax - edgemin)/bilateralRangeSteps) );
%     bsim(1:npix/2,npix/2+1:npix) = single( bilateralFilter(double(simcudanorm(1:npix/2,npix/2+1:npix)),double(simcudanorm(1:npix/2,npix/2+1:npix)),edgemin, edgemax, bilateralSigma, (edgemax - edgemin)/bilateralRangeSteps) );
%     bsim(npix/2+1:npix,1:npix/2) = single( bilateralFilter(double(simcudanorm(npix/2+1:npix,1:npix/2)),double(simcudanorm(npix/2+1:npix,1:npix/2)),edgemin, edgemax, bilateralSigma, (edgemax - edgemin)/bilateralRangeSteps) );
%     bsim(npix/2+1:npix,npix/2+1:npix) = single( bilateralFilter(double(simcudanorm(npix/2+1:npix,npix/2+1:npix)),double(simcudanorm(npix/2+1:npix,npix/2+1:npix)),edgemin, edgemax, bilateralSigma, (edgemax - edgemin)/bilateralRangeSteps) );

    L = bsim > (1+edgeZScore*Ssim);
    
    %extract initial edges
    L2 = imerode(imdilate(L,ones(3)),ones(3));
    edges2 = L2 - imerode(imerode(imerode(L2,ones(3)),ones(3)),ones(3)); %two-pixel edge width to work with
    rp = regionprops(logical(edges2),'BoundingBox','Area');
    
    %get rid of single-point particles, can't distinguish from noise or get
    %reliable size.
    areas = [rp.Area];
    rp(areas<=2) = [];
    
    %search out bboxes inside other bboxes (focus spots)
    bb = [rp.BoundingBox];
    bbxmin = bb(1:4:end); bbymin = bb(2:4:end);
    bbxmax = bbxmin + bb(3:4:end);
    bbymax = bbymin + bb(4:4:end);
    nb = numel(bb)/4;
    bbinside = zeros(nb,1);
    for m=1:nb
        for k=1:nb
            if ( bbxmin(m) >= bbxmin(k)) && (bbxmax(m)<=bbxmax(k)) && (bbymin(m)>=bbymin(k)) && (bbymax(m)<=bbymax(k)) && (m~=k)
                bbinside(m)=1;
            end
        end
    end
    %get rid of bad bounding boxes
    bbinside = logical(bbinside);
    rp(bbinside) = [];
    bbxmin(bbinside) = [];
    bbymin(bbinside) = [];
    bbxmax(bbinside) = [];
    bbymax(bbinside) = [];
    
    %adjust the bb's so that they're inside the image indices
    bbxmin = max(1,floor(bbxmin));
    bbymin = max(1,floor(bbymin));
    bbxmax = min(npix,ceil(bbxmax));
    bbymax = min(npix,ceil(bbymax));
    
    %estimate the depth of each particle
    nb = numel(rp);
    zestsimidx = zeros(nb,1);
    zestrminidx = zeros(nb,1);
    for k=1:nb
        wmask = edges2(bbymin(k):bbymax(k),bbxmin(k):bbxmax(k)) .* ...
            double(simcuda(bbymin(k):bbymax(k),bbxmin(k):bbxmax(k)));
        sidxest = wmask.*double(sidxcuda(bbymin(k):bbymax(k),bbxmin(k):bbxmax(k)));
        zestsimidx(k) = sum(sidxest(:))/sum(wmask(:));
    end
    zestsim = interp1(1:numel(zpl),zpl,zestsimidx);
   
    
    %reconstruct images of the particles
    [simgs, startidx, height, width] = holoExtractBBox_cuda(5 + C, zestsim/1000, [rp.BoundingBox], blobimgmargin, 658e-9, 6.8e-6, powOrder, .001);
    
    %prep some storage for the s, so images of particles
    spimg = zeros(size(simgs),'single');
    sopimg = zeros(size(simgs),'single');
    
    
    %estimate the sizes
    xposinbb = zeros(nb,1);
    yposinbb = zeros(nb,1);
    circ = zeros(nb,1);
    radplanar = zeros(nb,1);
    fmrad = zeros(nb,1);
    %roverest = 1; %defined in config section: number of pixels that my algorithm overestimates the radius

    xposinbbpc = zeros(nb,1);
    yposinbbpc = zeros(nb,1);
    circpc = zeros(nb,1);
    radplanarpc = zeros(nb,1);
    magpc = zeros(nb,1);
    
    meanintblob = zeros(nb,1);
    meanintbkg = zeros(nb,1);
    stdintblob = zeros(nb,1);
    stdintbkg = zeros(nb,1);
    meansim = zeros(nb,1);

    for j=1:nb
        blobimg = getParticleFromSeq(simgs, startidx, height, width, j);
        sb = size(blobimg);

        cannyedges = edge(blobimg);

        %edge points via steerables
        [s,so] = filterSteerable(blobimg,2);
        spimg(startidx(j):startidx(j)+width(j)*height(j)-1) = s(:);
        sopimg(startidx(j):startidx(j)+width(j)*height(j)-1) = so(:);
        
        ncircpts = max(4, ceil(1.5*pi*(sqrt(rp(j).BoundingBox(3)*rp(j).BoundingBox(4)))));
        %note: the 1.5 factor is a helper to accomodate thicker edges, more
        %gradient information due to smoothing
        intlist = sort(s(:),'descend');
        v = intlist(ncircpts);
        s_edge = s>=v;

        edgeimg = logical( (s_edge + cannyedges)>=1 );

        %use a random algorithm to estimate the particle size
        x = 1:sb(2);
        y = 1:sb(1);
        [X,Y] = meshgrid(x,y);
        [xc,yc,r,fm] = circCentSteerable(X(edgeimg),Y(edgeimg),...
            s(edgeimg),so(edgeimg),3*ncircpts);
        
       
        cmom = circularMoments(blobimg, xc, yc, r);
        [~,ceig] = eig(cmom);
        %circularity = ceig(1)/ceig(4); %min,max always
        eccentricity = sqrt(1 - ceig(1)/ceig(4));
        circularity = 1 - eccentricity; %my definition here.
        
        
        %also try using a phase-coded annulus to detect the circles
        [xp,yp,rpc,magp] = phasecodedDetection(s,[],[],'log','scaled'); %keep the defaults
        cmompc = circularMoments(blobimg, xp, yp, rpc);
        [~,ceig] = eig(cmompc);
        circularitypc = 1 - sqrt(1 - ceig(1)/ceig(4));
        
        %save the results for later
        circ(j) = circularity;
        xposinbb(j) = xc;
        yposinbb(j) = yc;
        radplanar(j) = r - roverestp;
        fmrad(j) = fm;
        
        %...and save the phase-coded estimated circle info
        xposinbbpc(j) = xp;
        yposinbbpc(j) = yp;
        magpc(j) = magp;
        circpc(j) = circularitypc;
        radplanarpc(j) = rpc;
        
        %calculate some intensity statistics
        blobmasksm = circimg([sb(2),sb(1)], radplanar(j)-1, xposinbb(j)-sb(2)/2, yposinbb(j)-sb(1)/2);
        blobmasklg = 1-circimg([sb(2),sb(1)], radplanar(j)+1, xposinbb(j)-sb(2)/2, yposinbb(j)-sb(1)/2);
        meanintblob(j) = sum(blobmasksm(:).*blobimg(:))/sum(blobmasksm(:));
        meanintbkg(j) = sum(blobmasklg(:).*blobimg(:))/sum(blobmasklg(:));
        stdintblob(j) = sqrt( sum(blobmasksm(:).*(blobimg(:) - meanintblob(j)).^2)/sum(blobmasksm(:)) );
        stdintbkg(j) = sqrt( sum(blobmasklg(:).*(blobimg(:) - meanintbkg(j)).^2)/sum(blobmasklg(:)) );
        meansim(j) = mean(s(edgeimg));
    end
   
     
    %back-calculate the diameter of the drop candidates
    zobjest = 1./(1./zestsim + 1./zsphhat);
    magfactor = zsphhat./(zsphhat - zobjest);
    diaest = pixelsize*2*radplanar./magfactor;
    diaestpc = pixelsize*2*radplanarpc./magfactor;
    
    
    
    %compute some additional metrics for the particle fits
    meanratio = meanintblob./meanintbkg;
    stdratio = stdintblob./stdintbkg;

    
    %compile data about the drops
    dropinfohdr = {'rp index','Diameter (um)',...
        'X position (mm)','Y position (mm)','Z position (mm)','Magnification',...
    	'Circularity','Mean SIM','Intensity Ratio','Std Ratio','Mean intensity',...
        'LDA value','P_drop','P_non','dia pc','x pc','y pc','circ pc','mag pc'};
    dropdatamatrix = [ [1:nb]', diaest, ...
        (bbxmin' + xposinbb - blobimgmargin)*pixelsize/1000, ...
        (bbymin' + yposinbb - blobimgmargin)*pixelsize/1000,...
        zobjest, magfactor, circ, meansim, ...
        meanratio, stdratio, meanintblob, ... %droplda, dropprob, bkgprob,...
        zeros(nb,3), ...
        diaestpc, ...
        (bbxmin' + xposinbbpc - blobimgmargin)*pixelsize/1000,...
        (bbymin' + yposinbbpc - blobimgmargin)*pixelsize/1000,...
        circpc, magpc];
    
    %calculate some LDA estimates
    ldaW = [0.01766, -1.1994, 6.0343, 1.7665, 33.6678, -0.4728, ...
        -9.7164, -0.0137, 0.2383, 0.6062]';
    droplda = dropdatamatrix(:,[2,6:11,15,18:19])*ldaW;
    dropprob = normcdf(droplda, -8.1356, 0.9452);
    bkgprob = 1 - normcdf(droplda, -9.7246, 0.8341);
    dropdatamatrix(:,12:14) = [droplda, dropprob, bkgprob];
    
    
    
    %look for multiple detections at the same location
    %(i was going to do a linked list, but i'm tired)
    xpos = round(saturate(xposinbb + bbxmin' - blobimgmargin, 1, npix));
    ypos = round(saturate(yposinbb + bbymin' - blobimgmargin, 1, npix));
    closebystuff = zeros([npix,npix],'uint8');
    for j=1:nb
        closebystuff(ypos(j),xpos(j)) = 1;
    end
    closepack = bwpack(closebystuff~=0);
    closebystuff = bwunpack( imdilate(closepack, strel('octagon',12),'ispacked'));
    closelabels = labelmatrix(bwconncomp(closebystuff));
    blobgrplabel = zeros(nb,1);
    for j=1:nb
        blobgrplabel(j) = closelabels(ypos(j),xpos(j));
    end
    ngroups = max(blobgrplabel);
    groupdatamatrix = zeros(nb,size(dropdatamatrix,2)+1); %the maximum size needed
    groupdatamatrix(1:ngroups,1) = 1:ngroups;
    clear rpgrp;
    extragroups = 0; %count of the number of additional groups, ie, when 
    %lateral grouping isn't good enough to describe the variation observed
    for j=1:ngroups
       idxingroup = find(blobgrplabel==j); %find the blobs which belong to this lateral grouping
       ningroup = numel(idxingroup);
       if ningroup==1
           subgroup{1} = idxingroup;
           nsubgroups = 1;
           %groupdatamatrix(j,2:end-1) = dropdatamatrix(idxingroup,2:end);
       else
           %figure out if there are more than one depth-based groups
           %start by compiling all the raw depth estimates using edge masks
           %(unweighted!)
           ne = zeros(ningroup,1); %keep track of the number of edge pixels
           for k=1:ningroup
               ne(k) = sum(sum(edges2(bbymin(idxingroup(k)):bbymax(idxingroup(k)),...
                   bbxmin(idxingroup(k)):bbxmax(idxingroup(k))) ));
           end
           endidx = cumsum(ne);
           startidx = [1; endidx(1:end-1)+1];
           sidxgroup = zeros(endidx(end),1);
           for k=1:ningroup
               edgeidx = find( edges2(bbymin(idxingroup(k)):bbymax(idxingroup(k)),...
                   bbxmin(idxingroup(k)):bbxmax(idxingroup(k))) );
               sidxsub = sidxcuda(bbymin(idxingroup(k)):bbymax(idxingroup(k)),...
                   bbxmin(idxingroup(k)):bbxmax(idxingroup(k)));
               sidxgroup(startidx(k):endidx(k)) = sidxsub(edgeidx);
           end
           zingroup = 1./(1./sidxgroup + 1./zsphhat); %convert index to object distances
           if std(zingroup)==0
               nmixtures = 1; 
           else
               [mix,optmix] = GaussianMixture(zingroup, ningroup,0,0);
               nmixtures = optmix.K; %number of mixtures in the optimum group
           end
           if nmixtures == 1
               subgroup{1} = idxingroup;
               nsubgroups = 1;
  
           else
               %decide which detections are in each subgroup;
               %for each subgroup, find the most probable mixture
               %membership
               Pnk = GMClassLikelihoodK(optmix,zingroup);
               [maxpnk, maxidx] = max(Pnk,[],2); %find the index of the mixture component
               %which maximizes P_nk
               
               %for each particle, figure out which mixture component it "belongs to" 
               pmixidx = zeros(ningroup,1);
               for p=1:ningroup
                   pmixidx(p) = mode(maxidx(startidx(p):endidx(p)));
               end
               
               %decide on the final number of subgroups
               mixidxused = unique(pmixidx); %which mixture components got used
               nsubgroups = numel(mixidxused); %number of subgroups
 
               for p=1:nsubgroups
                   subgroup{p} = idxingroup(logical(pmixidx==mixidxused(p)));
               end
           end
       end
       
       %rough guess for now: just average all the junk together.
       for k=1:nsubgroups
           if k==1
               storeidx = j;
           else
               extragroups = extragroups + 1;
               storeidx = ngroups+extragroups;
           end
           groupdatamatrix(storeidx,2:end-1) = mean(dropdatamatrix(subgroup{k},2:end),1);
           if numel(subgroup{k})>1
                groupdatamatrix(storeidx,end) = 1; %signify that this is a group of detections
           else
               groupdatamatrix(storeidx,end) = 0;
           end
           gbbxmin = min(bbxmin(subgroup{k}));
           gbbymin = min(bbymin(subgroup{k}));
           gbbxmax = max(bbxmax(subgroup{k}));
           gbbymax = max(bbymax(subgroup{k}));
           rpgrp(storeidx).BoundingBox = [gbbxmin, gbbymin, gbbxmax-gbbxmin, gbbymax-gbbymin];

       end
    end
    
    ngroups = ngroups + extragroups;
    groupdatamatrix = groupdatamatrix(1:ngroups,:);
    
    
    
    %save out data!
    %dropdatamatrix = [dropdatamatrix(:,1), blobgrplabel, dropdatamatrix(:,2:end)];
    %dropinfohdr = {dropinfohdr{1}, 'Group index', dropinfohdr{2:end}};
    %
    %replace the individual detections with the grouped detections
    dropdatamatrix = groupdatamatrix;
%     dropinfohdr = {'rp index','Diameter (um)',...
%         'X position (mm)','Y position (mm)','Z position (mm)','Magnification',...
%     	'Circularity','Mean SIM','Intensity Ratio','Std Ratio','Mean intensity',...
%         'LDA value','P_drop','P_non','dia pc','x pc','y pc','circ pc','mag pc',...
%         'Grouped detections (T/F)'}; 
    dropinfohdr = [dropinfohdr, 'Grouped detections (T/F)'];
    %dropdatamatrix = flipud(sortrows(dropdatamatrix,11)); %sort in descending LDA value
    sizedropmatrix = size(dropdatamatrix);
%     writeCSV([datadir,filesep,'results',filesep,'diameters_',num2str(holonamenum(bkg1idx(pnum))),'.csv'],...
%        dropinfohdr, flipud(sortrows(dropdatamatrix,11)) );
   writeCSV([datadir,filesep,'results',filesep,'diameters_',num2str(holonamenum(diveidx(pnum))),'.csv'],...
       dropinfohdr, flipud(sortrows(dropdatamatrix,12)) );
 
    
    
    %save out some additional data for later analysis
    rporig = rp;
    rp = rpgrp; %replace the rp with the grouped rps
%     save([datadir,filesep,'results',filesep,'diameters_H',...
%         num2str(holonamenum(bkg1idx(pnum))),'.mat'],...
%         'dropinfohdr','dropdatamatrix','sizedropmatrix');
    save([datadir,filesep,'results',filesep,'diameters_H',...
        num2str(holonamenum(diveidx(pnum))),'.mat'],...
        'dropinfohdr','dropdatamatrix','sizedropmatrix','rp');

%     save([datadir,filesep,'results',filesep,'particleImages_H',...
%         num2str(holonamenum(bkg1idx(pnum))),'.mat'],...
%         'simgs','startidx','height','width','rp','blobimgmargin',...
%         'xposinbb','yposinbb','simcuda','sidxcuda','rmincuda','circ','radplanar',...
%         'zpl','zsphhat','zestsim','powOrder'); %enough to reconstruct 'em if you've got C
%     save([datadir,filesep,'results',filesep,'particleImages_H',...
%         num2str(holonamenum(bkg1idx(pnum))),'.mat'],...
%         'simgs','startidx','height','width','blobimgmargin','rporig','blobgrplabel',...
%         'xposinbb','yposinbb','simcuda','sidxcuda','rmincuda','circ','radplanar',...
%         'zpl','zsphhat','zestsim','powOrder'); %enough to reconstruct 'em if you've got C
%     save([datadir,filesep,'results',filesep,'particleImages_H',...
%         num2str(holonamenum(diveidx(pnum))),'.mat'],...
%         'simgs','startidx','height','width','blobimgmargin','rporig','blobgrplabel',...
%         'xposinbb','yposinbb','simcuda','sidxcuda','rmincuda','circ','radplanar',...
%         'zpl','zsphhat','zestsim','powOrder'); %enough to reconstruct 'em if you've got C


    
    %create a plot of the results
    figure(1); clf;
    subplot(1,2,1);
    imagesc(rmincuda); 
    axis image; colormap(gray);
%     title(['R_{min}, H: ', num2str(holonamenum(bkg1idx(pnum))),...
%         ', ',num2str(adjtime(bkg1idx(pnum),4)),'h',...
%         num2str(adjtime(bkg1idx(pnum),5)),'m',...
%         num2str(adjtime(bkg1idx(pnum),6)),'s, ',...
%         'depth: ',num2str(round(holodepth(bkg1idx(pnum))))]);
	title(['R_{min}, H: ', num2str(holonamenum(diveidx(pnum))),...
        ', ',num2str(adjtime(diveidx(pnum),4)),'h',...
        num2str(adjtime(diveidx(pnum),5)),'m',...
        num2str(adjtime(diveidx(pnum),6)),'s, ',...
        'depth: ',num2str(round(holodepth(diveidx(pnum))))]);

    subplot(1,2,2);
    imagesc(simcuda);
    axis image;
    title(['SIM image, ',num2str(ngroups),' detected particles']);
    hold on;
    for j=1:ngroups
%        p = plotcircle(bbxmin(j) + xposinbb(j) - blobimgmargin,...
 %           bbymin(j) + yposinbb(j) - blobimgmargin,...
  %          radplanar(j));
        p = plotcircle(dropdatamatrix(j,3)*1000/pixelsize, ...
            dropdatamatrix(j,4)*1000/pixelsize,...
            dropdatamatrix(j,2)/dropdatamatrix(j,6)/pixelsize/2,20);
        if (dropprob(j)>bkgprob(j))
            set(p,'color',[0,circ(j),0]);
        else
            set(p,'color',[abs(circ(j)),0,0]);
        end
        %add the phase-coded results
        p = plotcircle(dropdatamatrix(j,16)*1000/pixelsize,...
            dropdatamatrix(j,17)*1000/pixelsize,...
            dropdatamatrix(j,15)*dropdatamatrix(j,6)/pixelsize/2,20);
        if dropdatamatrix(j,18)>.95
            set(p,'color',[1,1,0])
        else
            set(p,'color',[1,0,1]);
        end
    end
    hold off;
    set(gcf,'Position',[150,150,900,500]);
%     saveas(gcf,[datadir,filesep,'results',filesep,'detections_H',...
%         num2str(holonamenum(bkg1idx(pnum))),'.png']);
    saveas(gcf,[datadir,filesep,'results',filesep,'detections_H',...
        num2str(holonamenum(diveidx(pnum))),'.png']);


   
   %check the particles found...
   %jack_fitz_check_zestsimScript;
   %imwrite(af,[datadir,filesep,'results',filesep,'particleImagesPreview_H',...
   %    num2str(holonamenum(bkg1idx(pnum))),'.png'],'png');
   %clear af focusimgs

   %get time information
   tictocs1(pnum) = toc;
   
   %give some feedback for when we'll be done
   if (pnum>5)
      meanproctime = mean(tictocs1(1:pnum)); %in seconds
      nowtime = now();
%       eta = nowtime + meanproctime*(numel(bkg1idx) - pnum)/(3600*24);
      eta = nowtime + meanproctime*(numel(diveidx) - pnum)/(3600*24);

      etavec = datevec(eta);
%       waitbar(pnum/(numel(bkg1idx)),wbh,['Finding particles; ETA of ',...
%           num2str(etavec(4),'%02u'),':',num2str(etavec(5),'%02u'),':',...
%           num2str(round(etavec(6)),'%02u')]);
        waitbar(pnum/(numel(diveidx)),wbh,['Finding particles; ETA of ',...
          num2str(etavec(4),'%02u'),':',num2str(etavec(5),'%02u'),':',...
          num2str(round(etavec(6)),'%02u')]);
   else
       %waitbar(pnum/(numel(bkg1idx)),wbh);
       waitbar(pnum/numel(diveidx), wbh);
   end
   
end

%get rid of the waitbar
delete(wbh);





%remove stationary particles from the results
jack_fitz_removeStationaryParticlesFromDir2;


% 
% 
% %compile all the results for the particle sizes

%first, figure out the size of the data storage required
dirdia = dir([datadir,filesep,'results',filesep,'diameters_H*.mat']);
nresults = numel(dirdia);
pperdatafile = zeros(nresults,1);
for j=1:nresults
    load([datadir,filesep,'results',filesep,dirdia(j).name],'sizedropmatrix');
    pperdatafile(j) = sizedropmatrix(1);
end
% 
%read in the results files
particledata = zeros(sum(pperdatafile),sizedropmatrix(2)+2);
pstartidx = [1; 1+cumsum(pperdatafile(1:end))];
for j=1:nresults
    load([datadir,filesep,'results',filesep,dirdia(j).name],'dropdatamatrix');
    diadatanum = str2num(dirdia(j).name(12:end-4));
    holoidx = find(diadatanum == holonamenum);
    particledata(pstartidx(j):pstartidx(j)+pperdatafile(j)-1,1) = diadatanum;
    particledata(pstartidx(j):pstartidx(j)+pperdatafile(j)-1,2) = holodepth(holoidx);
    particledata(pstartidx(j):pstartidx(j)+pperdatafile(j)-1,3:sizedropmatrix(2)+2) = dropdatamatrix;
    %particledata(pstartidx(j):pstartidx(j)+pperdatafile(j)-1,3:sizedropmatrix(2)+2) = dropdatamatrix(:,[2,7,8,9,10,11]);
end


%load up a classifier
ctree = dlmread('ctree_dive5.txt','',1,1); %careful, this is currently in my numerics directory
classest = treeclassifier(particledata,ctree);
classnames = {'drop','non-drop'};

% %save the compiled results.
particleinfohdr = ['Hologram number','Depth (ft)',dropinfohdr];
save([datadir,filesep,'detected_particles.mat'],'particleinfohdr','particledata','classest','classnames');

%     
%filter and plot as you like.
jack_fitz_sizeDistributionEstPlots; %...for example.

%save a few samples for ground-truthing and getting particle size
%information (improve the detection/recognition and correct sizes)
jack_fitz_saveBkgSubSamples_script;