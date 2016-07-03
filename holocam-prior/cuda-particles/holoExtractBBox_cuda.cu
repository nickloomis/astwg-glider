
//reconstructs a hologram on a number of planes, saving out images of specific bounding boxes
//



#include "mex.h"
#include "cuda_runtime.h"
#include "cufft.h"






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
	return tempResult;
}




//multiply a complex value by a scale factor
__device__ cufftComplex complexScale32(cufftComplex a, float sc)
{
	cufftComplex tempResult;
	tempResult.x = a.x*sc;
	tempResult.y = a.y*sc;
	return tempResult;
}



__device__ float complexMagnitude(float2 a)
{
	return sqrtf( a.x*a.x + a.y*a.y );
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


static __global__ void copyBBKernel(float *bbdev, float2 *recdev, const int bbidx, const int width, const int height, 
									const int xstart, const int ystart, const int holoheight){
	int xidx = threadIdx.x + blockDim.x*blockIdx.x;
	int yidx = threadIdx.y + blockDim.y*blockIdx.y;
	if (xidx<width && yidx<height){
		int idx = xidx*height + yidx + bbidx;
		bbdev[idx] = complexMagnitude(recdev[(xidx + xstart)*holoheight + (yidx + ystart)]);
		//bbdev[idx] = .2f;
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





void reconstruct(float *bbimg, float *holosrc, double *z, const int nz, const int M, const int N, const int ntotalpix, 
				 const double lambda, const double pixelsize, 
				const double powOrder, const double powOffset, 
				int *bbxmin, int *bbymin, int *bbxmax, int *bbymax)
{
	float *holodev, *bbdev;
	float2 *cholodev, *recdev;
	float adudu, cutoff2, fftscale;
	float2 offsetPhaseExp;
	float sigma;
	int width, height, totalpix = 0;

	dim3 myblock(16, 16);
	dim3 mygrid( roundUpDiv(N,16), roundUpDiv(M,16) );
	dim3 copygrid(1,1);

	//upload the hologram and convert it to a complex-valued matrix
	cudaMalloc( (void**)&holodev, sizeof(float)*M*N);
	cudaMalloc( (void**)&cholodev, sizeof(float2)*M*N);
	cudaMemcpy( holodev, holosrc, sizeof(float)*M*N, cudaMemcpyHostToDevice);
	float_to_complexKernel<<<mygrid, myblock>>>(cholodev, holodev, M, N);
	cudaFree(holodev);

	//create an FFT plan
	cufftHandle plan;
	cufftPlan2d(&plan, N, M, CUFFT_C2C);

	//take the FFT of the hologram data
	cufftExecC2C(plan, cholodev, cholodev, CUFFT_FORWARD);
	

	//apply the power kernel filter
	if (powOrder > 0) {
		//mexPrintf("Applying power kernel\n");
		sigma = -1.0f/log(powOffset);
		powerKernel<<<mygrid, myblock>>>(cholodev, N, M,
			powOrder, powOffset, sigma);
	}

	//reserve memory for the reconstruction
	cudaMalloc( (void**)&recdev, sizeof(float2)*M*N);
	cudaMalloc( (void**)&bbdev, sizeof(float)*ntotalpix);


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

		//copy the bb portion of the image
		width = (bbxmax[i] - bbxmin[i] + 1);
		height = (bbymax[i] - bbymin[i] + 1);
		copygrid.x = roundUpDiv(width,16);
		copygrid.y = roundUpDiv(height,16);
		copyBBKernel<<<copygrid, myblock>>>(bbdev, recdev, totalpix, width, height, 
			bbxmin[i] - 1, bbymin[i] - 1, M);
		totalpix += width*height;

	}

	//return the data to the host
	cudaMemcpy(bbimg, bbdev, sizeof(float)*ntotalpix, cudaMemcpyDeviceToHost);

	//clean up!
	cufftDestroy(plan);
	cudaFree(cholodev);
	cudaFree(recdev);
	cudaFree(bbdev);
}



// [imgs,idxstart,height,width] = holoExtractBBox_cuda(img, z, BBarray, margin, lambda, pixelsize, ...
//           powerKernOrder, powerKernOffset);
// 
// where BBarray = [regionpropsstruct.BoundingBox]; (and it's a double by matlab defaults)	
//
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
	
	float *holosrc, *bbimg;
	double *dsrc, *bbarray;
	int *bbxmin, *bbxmax, *bbymin, *bbymax;
	double *width, *height, *idxstart;
	unsigned char *isrc;
	double *z, lambda = 658e-9, pixelsize = 9e-6;
	int M, N, nz, nb, margin = 0;
	float powOrder = -1, powOffset = .001;


	
	//get the size of the hologram
	M = mxGetM(prhs[0]); 
	N = mxGetN(prhs[0]); 
	

	//retrieve the depths at which to do reconstructions
	z = mxGetPr(prhs[1]);
	nz = mxGetM(prhs[1]) * mxGetN(prhs[1]);


	//get the boundingbox array
	bbarray = mxGetPr(prhs[2]);
	nb = mxGetM(prhs[2]) * mxGetN(prhs[2]) / 4;
	
	//the additional margin


	//get optional parameters; use defauls otherwise
	if (nrhs>=4)
		if (!mxIsEmpty(prhs[3]))
			margin = (int)mxGetScalar(prhs[3]);
	if (nrhs>=5)
		if (!mxIsEmpty(prhs[4]))
			lambda = mxGetScalar(prhs[4]);
	if (nrhs>=6)
		if (!mxIsEmpty(prhs[5]))
			pixelsize = mxGetScalar(prhs[5]);
	if (nrhs>=7)
		if (!mxIsEmpty(prhs[6]))
			powOrder = (float) mxGetScalar(prhs[6]);
	if (nrhs>=8)
		if (!mxIsEmpty(prhs[7]))
			powOffset = (float) mxGetScalar(prhs[7]);
	

	//figure out the size of each picture
	plhs[1] = mxCreateNumericMatrix(nb, 1, mxDOUBLE_CLASS, mxREAL);
	plhs[2] = mxCreateNumericMatrix(nb, 1, mxDOUBLE_CLASS, mxREAL);
	plhs[3] = mxCreateNumericMatrix(nb, 1, mxDOUBLE_CLASS, mxREAL);
	idxstart = mxGetPr(plhs[1]);
	height = mxGetPr(plhs[2]);
	width = mxGetPr(plhs[3]);

	bbxmin = (int*) malloc( sizeof(int) * nb);
	bbxmax = (int*) malloc( sizeof(int) * nb);
	bbymin = (int*) malloc( sizeof(int) * nb);
	bbymax = (int*) malloc( sizeof(int) * nb);
	//pixperbb = (int*) malloc( sizeof(int) * nb);
	int totalpix = 0;

	for (int i=0; i<nb; i++) {
		bbxmin[i] = max(1.0,(int)floor(bbarray[i*4]) - (double)margin );
		bbymin[i] = max(1.0,(int)floor(bbarray[i*4+1]) - (double)margin );
		bbxmax[i] = min((double)N, ceil(bbxmin[i] + bbarray[i*4+2] + (double)(margin*2) ));
		bbymax[i] = min((double)M, ceil(bbymin[i] + bbarray[i*4+3] + (double)(margin*2) ));
		width[i] = (double)(bbxmax[i] - bbxmin[i] + 1.0);
		height[i] = (double)(bbymax[i] - bbymin[i] + 1.0);
		//pixperbb[i] = (int)(width[i] * height[i]);
		idxstart[i] = totalpix+1; //the +1 is for matlab's fortran-style indexing.
		totalpix += (int)(width[i] * height[i]);
	}

	//mexPrintf("totalpix: %i\n", totalpix);
	

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
	
	//create an output vector to store the images
	plhs[0] = mxCreateNumericMatrix(totalpix, 1, mxSINGLE_CLASS, mxREAL);
	bbimg = (float*) mxGetPr(plhs[0]);


	//reconstruct hologram and find mins
	reconstruct(bbimg, holosrc, z, nz, M, N, totalpix, lambda, pixelsize, 
		powOrder, powOffset, bbxmin, bbymin, bbxmax, bbymax);

	//clean up
	if ( imgclass == mxDOUBLE_CLASS || imgclass == mxUINT8_CLASS )
		mxFree(holosrc);

	free(bbxmin);
	free(bbymin);
	free(bbxmax);
	free(bbymax);
	
}