%load an image
h = readDNG([datadir, filesep, filename]);

%subtract the background
S = bkgsubBayer(h, bkg, meanbkg);

%crop out the center
npix = 2048;
C = cropSmaller(S, npix);

%reconstruct minIntensity (rem: use a positive offset, power kernel, and
%spherical source)
zmin = 40; %in mm
%zmax = 40 + 11.75*25.4/1.33; %in mm, in water
zmax = 40 + 11*25.4/1.33;
zsphhat = 387; %in mm
zobj = zmin:.2:zmax;
zpl = 1./(1./zobj - 1/zsphhat);
powOrder = 14;
%[Rmin,Ridx] = minIntensityHoloCuda(C + min(C(:)) + 1e3, zpl/1000, ...
%    658e-9, 6.8e-6, powOrder);

% C : original hologram data
% zpl : vector reconstruction planes
%
[simcuda, sidxcuda, rmincuda, ridxcuda] = maxSIMfm(C, zpl/1000, 658e-9, 6.8e-6, 12, .001, 2, 500);
%takes ~5s for numel(z)=106 and npix=2048
%takes ~30s for numel(z)=106 and npix=4096
%takes ~45s for numel(z)=1059 and npix=2048
%should take ~5m for numel(z)=1059 and npix=4096! (but ~300x faster than
%pure matlab)

[M,S] = localstatsfilt_cuda(simcuda,150);
simcudanorm = simcuda./M; 
S = S./M;
edgemin = double(min(simcudanorm(:)));
edgemax = double(max(simcudanorm(:)));
bsim = bilateralFilter(double(simcudanorm),double(simcudanorm),edgemin, edgemax, 50, (edgemax - edgemin)/10);

%[M,S] = localstatsfilt_cuda(bsim,150);
L = bsim > (1+3*S);
%normalize to the background (so that bilateral, for example, works OK with
%the reconstructed images)
%calculate max(steerable*magnitude) for each reconstruction
%use bilateral filtering to smooth out the noise
%use localstatsfilt(bsim,100) to get the local stats
%threshold using (bsim>(M+3*S)) or similar
%dilate-erode the thresholded image
%erode-subtract to get edge pixels (note: need to completely close the
%droplets, or eliminate edges within the droplets)
%use steerable*magnitude weights, depth consistency, and steerable 
%direction to find diamter, circularity
%sort in order of an overall fit quality (mean edge pixel weights, ratio of
%edge pixels found to possible number of edge pixels, circularity, etc)
%store the data!

%bsim =
%bilateralFilter(double(SIM),double(SIM),double(min(SIM)),double(max(SIM)),
%something,something);

%[M,S] = localstatsfilt(bsim,100);
%L = bsim > (M+3*S);
L2 = imerode(imdilate(L,ones(3)),ones(3));
edges2 = L2 - imerode(imerode(imerode(L2,ones(3)),ones(3)),ones(3)); %two-pixel edge width to work with
%labels = bwlabel(edges2);
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
for j=1:nb
    for k=1:nb
        if ( bbxmin(j) >= bbxmin(k)) && (bbxmax(j)<=bbxmax(k)) && (bbymin(j)>=bbymin(k)) && (bbymax(j)<=bbymax(k)) && (j~=k)
            bbinside(j)=1;
        end
    end
end
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
for j=1:nb
    wmask = edges2(bbymin(j):bbymax(j),bbxmin(j):bbxmax(j)) .* ...
        double(simcuda(bbymin(j):bbymax(j),bbxmin(j):bbxmax(j)));
    sidxest = wmask.*double(sidxcuda(bbymin(j):bbymax(j),bbxmin(j):bbxmax(j)));
    ridxest = wmask.*double(ridxcuda(bbymin(j):bbymax(j),bbxmin(j):bbxmax(j)));
    zestsimidx(j) = sum(sidxest(:))/sum(wmask(:));
    zestrminidx(j) = sum(ridxest(:))/sum(wmask(:));
end
zestsim = interp1(1:numel(zpl),zpl,zestsimidx);
zestrmin = interp1(1:numel(zpl),zpl,zestrminidx);



%reconstruct images of the particles for estimating proper sizes

% simgs : all extracted ROI images, concatenated into a single array
% startidx : vector of starting indices
% height : vector of heights, one entry for each ROI
% width : vector of widths, one entry for each ROI
%
% Example use: get the pixels for the third image
%  num_pixels = height(3) * width(3);
%  end_index = startidx(3) + num_pixels - 1;
%  image_pixels_vector = simgs(startidx(3) : end_index);
%  roi = reshape(image_pixels_vector, [height(3), width(3)])

% C: original hologoram
margin = 10;
[simgs, startidx, height, width] = ...
  holoExtractBBox_cuda(C, zestsim/1000, [rp.BoundingBox], margin, 658e-9, 6.8e-6, powOrder, .001);


%estimate the sizes

xposinbb = zeros(nb,1);
yposinbb = zeros(nb,1);
circ = zeros(nb,1);
radplanar = zeros(nb,1);
fmrad = zeros(nb,1);
roverest = 1; %number of pixels that my algorithm overestimates the radius

meanintblob = zeros(nb,1);
meanintbkg = zeros(nb,1);
stdintblob = zeros(nb,1);
stdintbkg = zeros(nb,1);

% Select oil-shaped images from the ROIs.
for j=1:nb
    blobimg = getParticleFromSeq(simgs, startidx, height, width, j);
    sb = size(blobimg);
    
    cannyedges = edge(blobimg);
    
    %edge points via steerables
    [s,so] = filterSteerable(blobimg,2);
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
    [veig,ceig] = eig(cmom);
    circularity = ceig(1)/ceig(4); %min,max always
    
    %save the results for later
    circ(j) = circularity;
    xposinbb(j) = xc;
    yposinbb(j) = yc;
    radplanar(j) = r - roverest;
    fmrad(j) = fm;
  
    %display the sucker
%     mydisp(blobimg); colormap(gray); hold on;
%     p = plotcircle(xc,yc,r);
%     set(p,'color','r','linewidth',2);
%     plot([xc,xc+veig(1,1)*r*circularity],[yc,yc+veig(2,1)*r*circularity],'y',...
%         [xc,xc+veig(2,1)*r],[yc,yc+veig(2,2)*r],'w');
%     title(['C: ',num2str(circularity)],'fontsize',12);
%     hold off;
%     
%     drawnow();
    %pause(.5);
    
    %blobmask = circimg([sb(2),sb(1)], radplanar(j), xposinbb(j)-sb(2)/2, yposinbb(j)-sb(1)/2);
     %calculate some intensity statistics for later use
    blobmasksm = circimg([sb(2),sb(1)], radplanar(j)-1, xposinbb(j)-sb(2)/2, yposinbb(j)-sb(1)/2);
    blobmasklg = 1-circimg([sb(2),sb(1)], radplanar(j)-1, xposinbb(j)-sb(2)/2, yposinbb(j)-sb(1)/2);
    meanintblob(j) = sum(blobmasksm(:).*blobimg(:))/sum(blobmasksm(:));
    meanintbkg(j) = sum(blobmasklg(:).*blobimg(:))/sum(blobmasklg(:));
    stdintblob(j) = sqrt( sum(blobmasksm(:).*(blobimg(:) - meanintblob(j)).^2)/sum(blobmasksm(:)) );
    stdintbkg(j) = sqrt( sum(blobmasklg(:).*(blobimg(:) - meanintbkg(j)).^2)/sum(blobmasklg(:)) );
 
    
end

meanratio = meanintblob./meanintbkg;
stdratio = stdintblob./stdintbkg;

blobfeats = [radplanar, circ, fmrad, meanratio, stdratio, circ./radplanar];
ldaW = [.183, 2.19, -.211, -2.63, .725, .629]';
droplda = blobfeats*ldaW;
dropprob = normcdf(droplda,2.034,.5177); %based on my LDA of some labeled drops
bkgprob = 1-normcdf(droplda,1.0466,.8480);
%with that particular LDA, the higher the value the better chance it's an
%oil drop and not some background junk.
%pdfs:
%dropprob = 1/sqrt(2*pi*.5177^2)*exp(-(droplda - 2.034).^2/(2*.5177^2));
%bkgprob = 1/sqrt(2*pi*.8480^2)*exp(-(droplda - 1.0466).^2/(2*.8480^2));



%re-adjust the bboxes
% 
% bb = [rp.BoundingBox];
% bbxmin = bb(1:4:end); 
% bbymin = bb(2:4:end);
% bbxmax = bbxmin + bb(3:4:end);
% bbymax = bbymin + bb(4:4:end);
% 
% bbxmin = max(1,floor(bbxmin - margin));
% bbymin = max(1,floor(bbymin - margin));
% bbxmax = min(npix,ceil(bbxmax + margin));
% bbymax = min(npix,ceil(bbymax + margin));
% 
% [rimgs, startidx, height, width] = holoExtractBBox_cuda(C, zestrmin/1000, [rp.BoundingBox], margin, 658e-9, 6.8e-6, 12, .001);
% [simgs, startidx, height, width] = holoExtractBBox_cuda(C, zestsim/1000, [rp.BoundingBox], margin, 658e-9, 6.8e-6, 2, .001);
% 
% 
% 
% sprev = zeros(npix);
% %sprev = rmincuda;
% rprev = zeros(npix);
% for j=1:nb
%     rimg = reshape(simgs(startidx(j):startidx(j)+width(j)*height(j)-1),[height(j),width(j)]);
%     sprev(bbymin(j):bbymin(j)+height(j)-1,bbxmin(j):bbxmin(j)+width(j)-1) = (rimg);
%     rimg = reshape(rimgs(startidx(j):startidx(j)+width(j)*height(j)-1),[height(j),width(j)]);
%     rprev(bbymin(j):bbymin(j)+height(j)-1,bbxmin(j):bbxmin(j)+width(j)-1) = (rimg);
% end
% mydisp(sprev);

%select out everything inside the bounding box, use for estimating circle
%shape