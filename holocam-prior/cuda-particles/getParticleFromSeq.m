function P = getParticleFromSeq(imgseq,startidx,height,width,idx)

P = reshape(imgseq(startidx(idx):startidx(idx) + height(idx)*width(idx)-1), ...
    [height(idx),width(idx)]);