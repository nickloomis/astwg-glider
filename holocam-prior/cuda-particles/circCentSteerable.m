function [XC,YC,R,fm] = circCentSteerable(X,Y,S,SO,niters)

xc = zeros(niters,1);
yc = zeros(niters,1);
w = zeros(niters,1);

npts = numel(S);
iters = 0;
while iters<niters
%for j=1:niters
    randidx = randperm(npts);
    m1 = tan(SO(randidx(1)));
    m2 = tan(SO(randidx(2)));
    if (m1~=m2)
        iters = iters+1;
        b1 = Y(randidx(1)) - X(randidx(1))*m1;
        b2 = Y(randidx(2)) - X(randidx(2))*m2;
        v1 = [cos(SO(randidx(1))),sin(SO(randidx(1)))];
        v2 = [cos(SO(randidx(2))),sin(SO(randidx(2)))];
        cp = abs(v2(2)*v1(1) - v2(1)*v1(2));

        xc(iters) = (b2 - b1)/(m1 - m2);
        yc(iters) = m1*X(randidx(1)) + b1;
        w(iters) = sqrt(S(randidx(1))*S(randidx(2)))*cp.^2;
    end
end

XC = sum(w.*xc)/sum(w);
YC = sum(w.*yc)/sum(w);

%estimate radius
r = sqrt((X(:) - XC).^2 + (Y(:) - YC).^2);
R = sum(S(:).*r)/sum(S(:));

%need to calculate a weighted std as a fit metric
xcvar = sum(w.*(xc - XC).^2)/sum(w);
ycvar = sum(w.*(yc - YC).^2)/sum(w);
rvar = sum(S(:).*(r - R).^2)/sum(S(:));
fm = sqrt(xcvar + ycvar + rvar);

