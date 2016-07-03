
//reconstructs a hologram on a number of planes and find the magnitude and (steerable) gradient 
//at each depth. the maximum st_grad * (1 - mag) is recorded as a focus metric.



#include "mex.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include "float.h"





void double_to_float(float *dst, double *src, const int n)
{
	for (int i=0; i<n; i++)
		dst[i] = (float)src[i];
}




void uint8_to_float(float *dst, unsigned char *src, const int n)
{
	for (int i=0; i<n; i++)
		dst[i] = (float)src[i];
}




//integer division, but forces a round up if there are leftovers (ie, 3/2 = 1, but roundUpDiv(3/2) = 2)
int roundUpDiv(int num, int denom){
	return (num/denom) + (!(num%denom)? 0 : 1);
}




__device__ cufftComplex complexMult32(cufftComplex a, cufftComplex b)
{
	cufftComplex tempResult;
	tempResult.x = a.x * b.x - a.y * b.y;
	tempResult.y = a.y * b.x + a.x * b.y;
	//tempResult.x = __fmul_rn(a.x,b.x) - __fmul_rn(a.y,b.y);
	//tempResult.y = __fadd_rn( __fmul_rn(a.y,b.x) , __fmul_rn(a.x,b.y));
	return tempResult;
}




//multiply a complex value by a scale factor
__device__ cufftComplex complexScale32(cufftComplex a, float sc)
{
	cufftComplex tempResult;
	tempResult.x = a.x*sc;
	tempResult.y = a.y*sc;
	//tempResult.x = __fmul_rn(a.x,sc);
	//tempResult.y = __fmul_rn(a.y,sc);
	return tempResult;
}



__device__ float complexMagnitude(float2 a)
{
	return sqrtf( a.x*a.x + a.y*a.y );
	//return __fsqrt_rn( __fmul_rn(a.x,a.x) + __fmul_rn(a.y,a.y) );
	//return __fsqrt_rn( a.x*a.x + a.y*a.y );
}


#include "propagation_kernels.cu" //note: included here because the file
//references complexScale32 and complexMult32... I didn't want to edit 
//propagation_kernels.cu, and I'm too lazy right now to include headers.
//note that this is currently terrible coding, you should include headers.
//





static __global__ void float_to_complexKernel(float2 *dst, float *src, const int M, const int N)
{
	int xidx = threadIdx.x + blockDim.x*blockIdx.x;
	int yidx = threadIdx.y + blockDim.y*blockIdx.y;
	if (xidx<N && yidx<M){
		int idx = yidx*N + xidx;
		dst[idx].x = src[idx];
		dst[idx].y = 0;
	}
}



/*
static __global__ void fillFloatKernel(float *dst, const float value, 
									   const int M, const int N)
{
	int xidx = threadIdx.x + blockDim.x*blockIdx.x;
	int yidx = threadIdx.y + blockDim.y*blockIdx.y;
	if (xidx<N && yidx<M){
		dst[yidx*N+xidx] = value;
	}
}
*/


static __global__ void scaleFloatKernel(float *dst, const float sc, const int M, const int N)
{
	int xidx = threadIdx.x + blockDim.x*blockIdx.x;
	int yidx = threadIdx.y + blockDim.y*blockIdx.y;
	if (xidx<N && yidx<M){
		int idx = yidx*N + xidx;
		dst[idx] = dst[idx]*sc;
	}
}



static __global__ void copyRealKernel(float *dst, float2 *csrc, const int M, const int N)
{
	int xidx = threadIdx.x + blockDim.x*blockIdx.x;
	int yidx = threadIdx.y + blockDim.y*blockIdx.y;
	if (xidx<N && yidx<M){
		int idx = yidx*N+xidx;
		dst[idx] = csrc[idx].x;
	}
}


static __global__ void copyMagnitudeKernel(float *dst, float2 *csrc, const int M, const int N)
{
	int xidx = threadIdx.x + blockDim.x*blockIdx.x;
	int yidx = threadIdx.y + blockDim.y*blockIdx.y;
	if (xidx<N && yidx<M){
		int idx = yidx*N+xidx;
		dst[idx] = complexMagnitude(csrc[idx]);
	}
}


//finds the larger of the two
static __global__ void compareRealKernel(float *rintdev, short *ridxdev, float *recdev,
										  const int i, const int M, const int N)
{
	int xidx = threadIdx.x + blockDim.x*blockIdx.x;
	int yidx = threadIdx.y + blockDim.y*blockIdx.y;
	if (xidx<N && yidx<M){
		int idx = yidx*N+xidx;
		float newreal = recdev[idx];
		float oldreal = rintdev[idx];
		rintdev[idx] = (newreal > oldreal ? newreal : oldreal);
		ridxdev[idx] = (newreal > oldreal ? (short)i : ridxdev[idx]);
	}
}


//finds the larger of the two; this is the overloaded version that stores the intensity at the max sim as well.
static __global__ void compareRealKernel(float *sim, short *sidxdev, float *simdev, float *rintdev, float2 *recdev,
										  const int i, const int M, const int N)
{
	int xidx = threadIdx.x + blockDim.x*blockIdx.x;
	int yidx = threadIdx.y + blockDim.y*blockIdx.y;
	if (xidx<N && yidx<M){
		int idx = yidx*N+xidx;
		float newreal = simdev[idx];
		float oldreal = sim[idx];
		sim[idx] = (newreal > oldreal ? newreal : oldreal);
		sidxdev[idx] = (newreal > oldreal ? (short)i : sidxdev[idx]);
		rintdev[idx] = (newreal > oldreal ? recdev[idx].x : rintdev[idx]);
	}
}




//finds the smaller of the two
static __global__ void compareReal2Kernel(float *rintdev, short *ridxdev, float2 *recdev,
										  const int i, const int M, const int N)
{
	int xidx = threadIdx.x + blockDim.x*blockIdx.x;
	int yidx = threadIdx.y + blockDim.y*blockIdx.y;
	if (xidx<N && yidx<M){
		int idx = yidx*N+xidx;
		float newreal = recdev[idx].x;
		float oldreal = rintdev[idx];
		rintdev[idx] = (newreal < oldreal ? newreal : oldreal);
		ridxdev[idx] = (newreal < oldreal ? (short)i : ridxdev[idx]);
	}
}


/*
static __global__ void compareIntensities(float *rintdev, short *ridxdev, float2 *recdev,
										  const int i, const int M, const int N)
{
	int xidx = threadIdx.x + blockDim.x*blockIdx.x;
	int yidx = threadIdx.y + blockDim.y*blockIdx.y;
	if (xidx<N && yidx<M){
		int idx = yidx*N+xidx;
		float newmag = complexMagnitude( recdev[idx] );
		float oldmag = rintdev[idx];
		rintdev[idx] = (newmag < oldmag ? newmag : oldmag);
		ridxdev[idx] = (newmag < oldmag ? (short)i : ridxdev[idx]);
	}
}
*/

static __global__ void computeMagKernel(float2 *src, const int M, const int N)
{
	int xidx = threadIdx.x + blockDim.x*blockIdx.x;
	int yidx = threadIdx.y + blockDim.y*blockIdx.y;
	if (xidx<N && yidx<M) {
		int idx = yidx*N+xidx;
		src[idx].x = complexMagnitude(src[idx]);
		src[idx].y = 0;
	}
}




static __global__ void complexMultKernel(float2 *dst, float2 *src1, float2 *src2, const int M, const int N)
{
	int xidx = threadIdx.x + blockDim.x*blockIdx.x;
	int yidx = threadIdx.y + blockDim.y*blockIdx.y;
	if (xidx<N && yidx<M) {
		int idx = yidx*N+xidx;
		dst[idx] = complexMult32(src1[idx], src2[idx]);
	}
}



/*
static __global__ void applySyKernel(float2 *src, float2 *sfiltx, const int M, const int N)
{
	int xidx = threadIdx.x + blockDim.x*blockIdx.x;
	int yidx = threadIdx.y + blockDim.y*blockIdx.y;
	if (xidx<N && yidx<M) {
		int idx = yidx*N+xidx;
		int idx_t = xidx*M + yidx; //TODO: check this if M!=N
		src[idx] = complexMult32(src[idx], sfiltx[idx_t]);
	}
}
*/


static __global__ void sfiltMagKernel(float *sfiltmag, float2 *sx, float2 *sy, const int M, const int N){
	int xidx = threadIdx.x + blockDim.x*blockIdx.x;
	int yidx = threadIdx.y + blockDim.y*blockIdx.y;
	if (xidx<N && yidx<M) {
		int idx = yidx*N + xidx;
		float sxmag = complexMagnitude(sx[idx]);
		float symag = complexMagnitude(sy[idx]);
		sfiltmag[idx] = sqrtf( sxmag * sxmag + symag * symag );
		//sfiltmag[idx] = __fsqrt_rn(sxmag * sxmag + symag * symag );
	}
}	



/*
//NB: this kernel assumes that the magnitude is stored in the x component of the float2 src!
static __global__ void localMinKernel(float *dst, float2 *src, const int M, const int N)
{
	int xidx = threadIdx.x + blockDim.x*blockIdx.x;
	int yidx = threadIdx.y + blockDim.y*blockIdx.y;
	if ( xidx<(N-1) && xidx>=1 && yidx<(M-1) && yidx>=1) //limits so that I don't have to do special junk for the edges, corners
	{
		int idx = yidx*N + xidx;
		float lmin = min(src[idx - N - 1].x, src[idx - 1].x);
		lmin = min(lmin, src[idx + N -1].x);
		lmin = min(lmin, src[idx - N].x);
		lmin = min(lmin, src[idx].x);
		lmin = min(lmin, src[idx + N].x);
		lmin = min(lmin, src[idx - N + 1].x);
		lmin = min(lmin, src[idx + 1].x);
		lmin = min(lmin, src[idx + N +1].x);
		dst[idx] = lmin;
	}
}
*/
//this is terrible, actually. use texture filtering if the 3x3 local neighborhood is actually useful!






static __global__ void computeSIMkernel(float *sim, float *sfiltmag, float *localmag, const float intmax, 
										const int M, const int N)
{
	int xidx = threadIdx.x + blockDim.x*blockIdx.x;
	int yidx = threadIdx.y + blockDim.y*blockIdx.y;
	if (xidx<N && yidx<M) {
		int idx = yidx*N + xidx;
		sim[idx] = sfiltmag[idx] * (intmax - localmag[idx]);
	}
}



static __global__ void addShortKernel(short *dst, const short value, const int M, const int N)
{
	int xidx = threadIdx.x + blockDim.x*blockIdx.x;
	int yidx = threadIdx.y + blockDim.y*blockIdx.y;
	if (xidx<N && yidx<M){
		dst[yidx*N+xidx] += value;
	}
}




//NB: the source and destination are the same location!
//TODO: this might be a good use for atomic operators.
static __global__ void powerKernel(float2 *fsrc, const int xpix, const int ypix,
								   const float pow, const float offset, const float sigma){
	int xidx = threadIdx.x + blockDim.x * blockIdx.x; //pixel index that this thread deals with
	int yidx = threadIdx.y + blockDim.y * blockIdx.y;

	if (xidx < xpix && yidx < ypix){

		//get the FFTW-shifted u,v coordinates (in numbers of samples)
		//option 4) use normalized freqs (just the magnitude) - note that uidx->u here in the nomenclature.
		int halfx = xpix/2, halfy = ypix/2;
		float u = (xidx >= halfx ? (float)(xpix - xidx)/(float)halfx : (float)xidx/(float)halfx);
		float v = (yidx >= halfy ? (float)(ypix - yidx)/(float)halfy : (float)yidx/(float)halfy);

		int idx = yidx * ypix + xidx; //the linear index of the data
		float2 value = fsrc[idx]; //the value to work with

		//do some operation on value using u,v
		//here, i'm using a power filter to keep the lower freqs but kill the high freqs.
		float fkeep = __expf( - ( __powf(u,pow) + __powf(v,pow) )/sigma);
		value.x = value.x * fkeep;
		value.y = value.y * fkeep; 
		//the offset is the fraction to keep no matter what; only (1-offset) of the value
		//is allowed to be changed.

		//store the result back to the destination
		fsrc[idx] = value;
	}
}

static __global__ void fillSfiltxKernel(float2 *sfiltx, const float sigma, 
														   const int xpix, const int ypix)
{
	int xidx = threadIdx.x + blockDim.x * blockIdx.x; //pixel index that this thread deals with
	int yidx = threadIdx.y + blockDim.y * blockIdx.y;

	if (xidx < xpix && yidx < ypix){
		int halfx = xpix/2, halfy = ypix/2;
		float u = (xidx >= halfx ? -(float)(xpix - xidx) : (float)xidx);
		float v = (yidx >= halfy ? -(float)(ypix - yidx) : (float)yidx);

		int idx = yidx * ypix + xidx; //the linear index of the data
		
		//set the value
		sfiltx[idx].y = 0;
		sfiltx[idx].x = -2*u*__expf(-(u*u + v*v)/(2*sigma*sigma));
	}
}



static __global__ void fillSfiltyKernel(float2 *sfiltx, const float sigma, 
														   const int xpix, const int ypix)
{
	int xidx = threadIdx.x + blockDim.x * blockIdx.x; //pixel index that this thread deals with
	int yidx = threadIdx.y + blockDim.y * blockIdx.y;

	if (xidx < xpix && yidx < ypix){
		int halfx = xpix/2, halfy = ypix/2;
		float u = (xidx >= halfx ? -(float)(xpix - xidx) : (float)xidx);
		float v = (yidx >= halfy ? -(float)(ypix - yidx) : (float)yidx);

		int idx = yidx * ypix + xidx; //the linear index of the data
		
		//set the value
		sfiltx[idx].y = 0;
		sfiltx[idx].x = -2*v*__expf(-(u*u + v*v)/(2*sigma*sigma));
	}
}



void reconstruct(float *SIM, short *Sidx, float *Rint, short *Ridx, float *holosrc, 
					 double *z, const int nz, const int M, const int N, 
					 const double lambda, const double pixelsize,
					 const float powOrder, const float powOffset,
					 const float sfiltsigma, const float intmax,
					 float *ratsim, bool rintmode, bool ratsimmode)
{
	float *holodev, *rintdev, *simmaxdev, *sfiltmag, *localmag, *sim, *ratsimdev;
	short *ridxdev, *sidxdev;
	float2 *cholodev, *recdev, *sfiltx, *sfilty, *sx, *sy;
	float adudu, cutoff2, fftscale;
	float2 offsetPhaseExp;
	float sigma;

	dim3 myblock(16, 16);
	dim3 mygrid( roundUpDiv(N,16), roundUpDiv(M,16) );

	mexPrintf("Uploading hologram\n");

	//upload the hologram and convert it to a complex-valued matrix
	cudaMalloc( (void**)&holodev, sizeof(float)*M*N);
	cudaMalloc( (void**)&cholodev, sizeof(float2)*M*N);
	cudaMemcpy( holodev, holosrc, sizeof(float)*M*N, cudaMemcpyHostToDevice);
	float_to_complexKernel<<<mygrid, myblock>>>(cholodev, holodev, M, N);
	cudaFree(holodev);

	mexPrintf("Creating memory on GPU\n");

	//create variables to track the max SIM, min int
	cudaMalloc( (void**)&simmaxdev, sizeof(float)*M*N);
	cudaMalloc( (void**)&sidxdev, sizeof(short)*M*N);
	cudaMemset(sidxdev, 0, sizeof(short)*M*N);
	//fillFloatKernel<<<mygrid, myblock>>>(rintdev, FLT_MIN, M, N); //don't need this
	if (rintmode){
		cudaMalloc( (void**)&rintdev, sizeof(float)*M*N);
		cudaMalloc( (void**)&ridxdev, sizeof(short)*M*N);
		cudaMemset(ridxdev, 0, sizeof(short)*M*N);
	}

	mexPrintf("Creating FFT plan\n");

	//create an FFT plan
	cufftHandle plan;
	cufftPlan2d(&plan, N, M, CUFFT_C2C);

	mexPrintf("Executing FFT\n");
	
	//compute the (power-modulated) spectra, which will be used for each reconstruction step
	//take the FFT of the hologram data
	cufftExecC2C(plan, cholodev, cholodev, CUFFT_FORWARD);

	mexPrintf("Applying power filter\n");

	//apply the power kernel filter
	if (powOrder > 0) {
		//mexPrintf("Applying power kernel\n");
		sigma = -1.0f/log(powOffset);
		powerKernel<<<mygrid, myblock>>>(cholodev, N, M,
			powOrder, powOffset, sigma);
	}

	mexPrintf("Computing steerable filter\n");

	//compute the steerable filter, used for each reconstruction step
	cudaMalloc((void**) &sfiltx, sizeof(float2)*M*N);
	cudaMalloc((void**) &sfilty, sizeof(float2)*M*N);
	fillSfiltxKernel<<<mygrid,myblock>>>(sfiltx,sfiltsigma,M,N);
	fillSfiltyKernel<<<mygrid,myblock>>>(sfilty,sfiltsigma,M,N);
	cufftExecC2C(plan, sfiltx, sfiltx, CUFFT_FORWARD); //compute the freq-domain version of the filter
	cufftExecC2C(plan, sfilty, sfilty, CUFFT_FORWARD); 
	//..and reserve memory for the resulting sx, sy gradients
	cudaMalloc((void**) &sx, sizeof(float2)*M*N);
	cudaMalloc((void**) &sy, sizeof(float2)*M*N);
	cudaMalloc((void**) &sfiltmag, sizeof(float)*M*N);



	//reserve memory for the reconstruction
	cudaMalloc( (void**)&recdev, sizeof(float2)*M*N);


	//reserve memory for the focus metric components
	cudaMalloc( (void**)&localmag, sizeof(float)*M*N);
	cudaMalloc( (void**)&sim, sizeof(float)*M*N);
	if (ratsimmode)
		cudaMalloc( (void**)&ratsimdev, sizeof(float)*M*N);

	//fillFloatKernel<<<mygrid, myblock>>>(localmag, 1.0f, M, N); //don't need this

	mexPrintf("Computing reconstructions\n");

	//reconstruct the hologram at each depth in z, compare it to the previous
	//minimum intensity
	for (int i=0; i<nz; i++){
		//copy the fft spectrum to a variable that's temporary for each plane
		cudaMemcpy(recdev, cholodev, sizeof(float2)*M*N, cudaMemcpyDeviceToDevice);

		//set the propagation parameters
		setPropagationParams( (float)(z[i]), (float)lambda, (float)pixelsize, M, N, 
			&adudu, &cutoff2, &offsetPhaseExp, &fftscale, false);

		//apply the propagation kernel (full... could speed this up by ignoring cutoff)
		fresnelKernel<<<mygrid, myblock>>>(recdev, M, N, adudu, 
			cutoff2, offsetPhaseExp, fftscale);

		//take the inverse transform: get an image at the plane.
		cufftExecC2C(plan, recdev, recdev, CUFFT_INVERSE);

		//compute the magnitude
		computeMagKernel<<<mygrid, myblock>>>(recdev, M, N);

		//compute the steerable gradient responses in the x and y directions
		cudaMemcpy( sx, recdev, sizeof(float2)*M*N, cudaMemcpyDeviceToDevice); //copy the magnitude info
		cufftExecC2C(plan, sx, sx, CUFFT_FORWARD); //the spectrum of the magnitude
		cudaMemcpy( sy, sx, sizeof(float2)*M*N, cudaMemcpyDeviceToDevice); //copy the spectrum information
		//apply the sfilt kernels in the x and y directions
		complexMultKernel<<<mygrid, myblock>>>(sx, sx, sfiltx, M, N);
		complexMultKernel<<<mygrid, myblock>>>(sy, sy, sfilty, M, N);
		cufftExecC2C(plan, sx, sx, CUFFT_INVERSE);
		cufftExecC2C(plan, sy, sy, CUFFT_INVERSE);
		//finally: compute the magnitude and orientation
		sfiltMagKernel<<<mygrid,myblock>>>(sfiltmag, sx, sy, M, N);
		scaleFloatKernel<<<mygrid,myblock>>>(sfiltmag, 1.0f/((float)(M*M) * (float)(N*N)), M, N);
		//TODO: add orientation later, if desired


		//find the local min intensity (for SIM)
		//localMinKernel<<<mygrid, myblock>>>(localmag, recdev, M, N);
		copyRealKernel<<<mygrid, myblock>>>(localmag, recdev, M, N);


		//compute the SIM metric
		computeSIMkernel<<<mygrid, myblock>>>(sim, sfiltmag, localmag, intmax, M, N);
		

		if (rintmode){
			//compare the current intensity to the previous minimum
			if (i==0) 
				copyRealKernel<<<mygrid,myblock>>>(rintdev, recdev, M, N);
			else 
				compareReal2Kernel<<<mygrid, myblock>>>(rintdev, ridxdev, recdev, i, M, N);
		}

		//compare the current SIM to the previous SIMs
		if (i==0) {
			cudaMemcpy(simmaxdev, sim, sizeof(float)*M*N, cudaMemcpyDeviceToDevice);
			if (ratsimmode)
				cudaMemcpy(ratsimdev, rintdev, sizeof(float)*M*N, cudaMemcpyDeviceToDevice);
		}
		else {
			if (!ratsimmode)
				compareRealKernel<<<mygrid, myblock>>>(simmaxdev, sidxdev, sim, i, M, N);
			else //ratsimmode==true
				compareRealKernel<<<mygrid, myblock>>>(simmaxdev, sidxdev, sim, ratsimdev, recdev, i, M, N);
		}


	}

	//convert from c-style indexing to matlab-style indexing
	if (rintmode)
		addShortKernel<<<mygrid, myblock>>>(ridxdev, 1, M, N);
	addShortKernel<<<mygrid, myblock>>>(sidxdev, 1, M, N);

	mexPrintf("Copying results back to host\n");

	//return the data to the host
	cudaMemcpy(SIM, simmaxdev, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(Sidx, sidxdev, sizeof(short)*M*N, cudaMemcpyDeviceToHost);
	if (rintmode){
		cudaMemcpy(Rint, rintdev, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
		cudaMemcpy(Ridx, ridxdev, sizeof(short)*M*N, cudaMemcpyDeviceToHost);
	}
	if (ratsimmode)
		cudaMemcpy(ratsim, ratsimdev, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

	mexPrintf("Cleaning up memory and plans\n");

	//clean up!
	cufftDestroy(plan);
	cudaFree(cholodev);
	cudaFree(recdev);
	if (rintmode){
		cudaFree(rintdev);
		cudaFree(ridxdev);
	}
	cudaFree(sfiltx);
	cudaFree(sfilty);
	cudaFree(sx);
	cudaFree(sy);
	cudaFree(sfiltmag);
	cudaFree(sim);
	cudaFree(localmag);
	cudaFree(sidxdev);
	cudaFree(simmaxdev);
	if (ratsimmode)
		cudaFree(ratsimdev);
}



// [SIM, Sidx, Rint, Ridx, R_at_SIM] = maxSIMfm(img, z, lambda, pixelsize, ...
//           powerKernOrder, powerKernOffset, sigma, intmax);
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
	
	float *holosrc, *Rint, *SIM, *ratsim = NULL;
	double *dsrc;
	unsigned char *isrc;
	short *Ridx, *Sidx;
	double *z, lambda = 658e-9, pixelsize = 9e-6;
	int M, N, nz;
	float powOrder = -1, powOffset = .001, sigma = 2, intmax = 1000;
	bool rintmode, ratsimmode;

	if (nrhs<2)
		mexErrMsgTxt("maxSIMfm(image, z, [lambda], [pixelsize]) requires at least two arguments.");

	//get the size of the hologram
	M = mxGetN(prhs[0]); //number of y-direction pixels
	N = mxGetM(prhs[0]); //number of x-direction pixels
	//NB: I'm mixing M, N so that they align with the Matlab imagesc plots.
	//M is the y-direction , N is the x-direction.
	//this is because Matlab stores its data reading down the rows (row-major), while
	//C is column-wise ordering (column-major)
	
	//mexPrintf("nrhs = %i\n", nrhs);

	//retrieve the depths at which to do reconstructions
	z = mxGetPr(prhs[1]);
	nz = mxGetM(prhs[1])*mxGetN(prhs[1]);
	
	//mexPrintf("M: %i, N: %i, nz: %i\n", M, N, nz);

	//get optional parameters; use defauls otherwise
	if (nrhs>=3)
		if (!mxIsEmpty(prhs[2]))
			lambda = mxGetScalar(prhs[2]);
	if (nrhs>=4)
		if (!mxIsEmpty(prhs[3]))
			pixelsize = mxGetScalar(prhs[3]);
	if (nrhs>=5)
		if (!mxIsEmpty(prhs[4]))
			powOrder = (float) mxGetScalar(prhs[4]);
	if (nrhs>=6)
		if (!mxIsEmpty(prhs[5]))
			powOffset = (float) mxGetScalar(prhs[5]);
	if (nrhs>=7)
		if (!mxIsEmpty(prhs[6]))
			sigma = (float) mxGetScalar(prhs[6]);
	if (nrhs>=8)
		if (!mxIsEmpty(prhs[7]))
			intmax = (float) mxGetScalar(prhs[7]);
			

	//get the holographic image! 
	// note: if it's double or int, need to convert to float.

	mxClassID imgclass = mxGetClassID(prhs[0]);
	switch (imgclass){
		case mxSINGLE_CLASS:
			holosrc = (float*) mxGetPr(prhs[0]);
			break;
		case mxDOUBLE_CLASS:
			dsrc = mxGetPr(prhs[0]);
			holosrc = (float*) mxMalloc( sizeof(float)*M*N );
			double_to_float(holosrc, dsrc, M*N);
			break;
		case mxUINT8_CLASS:
			isrc = (unsigned char*) mxGetPr(prhs[0]);
			holosrc = (float*) mxMalloc( sizeof(float)*M*N );
			uint8_to_float(holosrc, isrc, M*N);
			break;
		default:
			mexErrMsgTxt("Input image needs to be single, double, or uint8.");
			break;
	}
	
	//create outputs
	
	if (nlhs>2)
		rintmode = true;
	else
		rintmode = false;
	if (nlhs>4)
		ratsimmode = true;
	else
		ratsimmode = false;

	plhs[0] = mxCreateNumericMatrix(N, M, mxSINGLE_CLASS, mxREAL); //the order is reversed here
	plhs[1] = mxCreateNumericMatrix(N, M, mxINT16_CLASS, mxREAL); //from what you'd expect... row-major vs column-major.
	SIM = (float*)mxGetPr(plhs[0]);
	Sidx = (short*)mxGetPr(plhs[1]);

	if (rintmode){
		plhs[2] = mxCreateNumericMatrix(N, M, mxSINGLE_CLASS, mxREAL);
		plhs[3] = mxCreateNumericMatrix(N, M, mxINT16_CLASS, mxREAL);
		Rint = (float*)mxGetPr(plhs[2]); //single
		Ridx = (short*)mxGetPr(plhs[3]); //int16
	}

	if (ratsimmode){
		plhs[4] = mxCreateNumericMatrix(N, M, mxSINGLE_CLASS, mxREAL);
		ratsim  = (float*)mxGetPr(plhs[4]);
	}

	//mexPrintf("Derivative: %i\n", derivOrder);

	//reconstruct hologram and find mins
	reconstruct(SIM, Sidx, Rint, Ridx, holosrc, z, nz, M, N, lambda, pixelsize, 
		powOrder, powOffset, sigma, intmax, ratsim, rintmode, ratsimmode);

	//clean up
	if ( imgclass == mxDOUBLE_CLASS || imgclass == mxUINT8_CLASS )
		mxFree(holosrc);
}