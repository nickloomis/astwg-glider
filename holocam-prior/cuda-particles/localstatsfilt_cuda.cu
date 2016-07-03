
// computes the local mean and standard deviation for an image.

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





static __global__ void complexMultiplyKernel(float2 *src, float2 *sfiltx, const int M, const int N)
{
	int xidx = threadIdx.x + blockDim.x*blockIdx.x;
	int yidx = threadIdx.y + blockDim.y*blockIdx.y;
	if (xidx<N && yidx<M) {
		int idx = yidx*N+xidx;
		src[idx] = complexMult32(src[idx], sfiltx[idx]);
	}
}



static __global__ void squareRealPartKernel(float2 *dst, float2 *src, const int M, const int N)
{
	int xidx = threadIdx.x + blockDim.x*blockIdx.x;
	int yidx = threadIdx.y + blockDim.y*blockIdx.y;
	if (xidx<N && yidx<M) {
		int idx = yidx*N+xidx;
		dst[idx].x = src[idx].x * src[idx].x;
	}
}



static __global__ void copyScaledRealPartKernel(float *dst, float2 *src, const float sc, const int M, const int N)
{
	int xidx = threadIdx.x + blockDim.x*blockIdx.x;
	int yidx = threadIdx.y + blockDim.y*blockIdx.y;
	if (xidx<N && yidx<M) {
		int idx = yidx*N+xidx;
		dst[idx] = src[idx].x * sc;
	}
}


static __global__ void computeStdKernel(float *sdev, float2 *cimgdevsq, float *mdev, 
										const float scale, const int M, const int N)
{
	int xidx = threadIdx.x + blockDim.x*blockIdx.x;
	int yidx = threadIdx.y + blockDim.y*blockIdx.y;
	if (xidx<N && yidx<M) {
		int idx = yidx*N+xidx;
		sdev[idx] = sqrtf(cimgdevsq[idx].x*scale - mdev[idx]*mdev[idx]);
	}
}



static __global__ void makeGaussianKernel(float2 *sfilt, const float sigma, 
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
		sfilt[idx].y = 0;
		sfilt[idx].x = __expf(-(u*u + v*v)/(2*sigma*sigma));
	}
}



void localstats(float *localmean, float *localstd, float *imgsrc, const int M, const int N, const float sigma)
{
	float *imgdev, *mdev, *sdev;
	float2 *cimgdev, *cimgdevsq, *gfilt;
	float scale;

	dim3 myblock(16, 16);
	dim3 mygrid( roundUpDiv(N,16), roundUpDiv(M,16) );

	//upload the hologram and convert it to a complex-valued matrix
	cudaMalloc( (void**)&imgdev, sizeof(float)*M*N);
	cudaMalloc( (void**)&cimgdev, sizeof(float2)*M*N);
	cudaMemcpy( imgdev, imgsrc, sizeof(float)*M*N, cudaMemcpyHostToDevice);
	float_to_complexKernel<<<mygrid, myblock>>>(cimgdev, imgdev, M, N);
	cudaFree(imgdev);
	
	//square the image values
	cudaMalloc( (void**)&cimgdevsq, sizeof(float2)*M*N);
	cudaMemcpy(cimgdevsq, cimgdev, sizeof(float2)*M*N, cudaMemcpyDeviceToDevice); //copy the raw image values
	squareRealPartKernel<<<mygrid, myblock>>>(cimgdevsq, cimgdevsq, M, N);


	//create an FFT plan
	cufftHandle plan;
	cufftPlan2d(&plan, M, N, CUFFT_C2C);


	//compute the Gaussian smoothing filter
	cudaMalloc((void**) &gfilt, sizeof(float2)*M*N);
	makeGaussianKernel<<<mygrid,myblock>>>(gfilt, sigma, M, N);
	cufftExecC2C(plan, gfilt, gfilt, CUFFT_FORWARD); //compute the freq-domain version of the filter


	//do the filtering

	//calculate the FFT of the image and the square image
	cufftExecC2C(plan, cimgdev, cimgdev, CUFFT_FORWARD);
	cufftExecC2C(plan, cimgdevsq, cimgdevsq, CUFFT_FORWARD);

	//multiply by the gaussian
	complexMultiplyKernel<<<mygrid, myblock>>>(cimgdev, gfilt, M, N);
	complexMultiplyKernel<<<mygrid, myblock>>>(cimgdevsq, gfilt, M, N);

	//take the inverse FFT
	cufftExecC2C(plan, cimgdev, cimgdev, CUFFT_INVERSE);
	cufftExecC2C(plan, cimgdevsq, cimgdevsq, CUFFT_INVERSE);


	//do the last steps of the statistics

	//copy out the real part and scale the values
	scale = 1.0f/((float)(M) * (float)(N)) * 1.0f/(6.28318531f*sigma*sigma);
	//scale = 1.0f;
	cudaMalloc((void**)&mdev, sizeof(float)*M*N);
	cudaMalloc((void**)&sdev, sizeof(float)*M*N);
	copyScaledRealPartKernel<<<mygrid, myblock>>>(mdev, cimgdev, scale, M, N);
	computeStdKernel<<<mygrid, myblock>>>(sdev, cimgdevsq, mdev, scale, M, N);
	

	//return the data to the host
	cudaMemcpy(localmean, mdev, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(localstd, sdev, sizeof(float)*M*N, cudaMemcpyDeviceToHost);


	//clean up!
	cufftDestroy(plan);
	cudaFree(mdev);
	cudaFree(sdev);
	cudaFree(gfilt);
	cudaFree(cimgdev);
	cudaFree(cimgdevsq);
	
}



// [M,S] = localstatsfilt(img,sigma);

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
	
	float *imgsrc, *localmean, *localstd;
	double *dsrc;
	unsigned char *isrc;
	int M, N;
	float sigma = 42.0f;
	mxClassID imgclass;

	if (nrhs<1)
		mexErrMsgTxt("localstatsfilt(img,sigma) requires at least an input image");

	//get the size of the image
	M = mxGetM(prhs[0]); //number of y-direction pixels
	N = mxGetN(prhs[0]); //number of x-direction pixels

	
	//get optional parameters; use defauls otherwise
	if (nrhs>=2)
		if (!mxIsEmpty(prhs[1]))
			sigma = (float) mxGetScalar(prhs[1]);


	//get the image, convert if not single
	// note: if it's double or int, need to convert to float.

	imgclass = mxGetClassID(prhs[0]);
	switch (imgclass){
		case mxSINGLE_CLASS:
			imgsrc = (float*) mxGetPr(prhs[0]);
			break;
		case mxDOUBLE_CLASS:
			dsrc = mxGetPr(prhs[0]);
			imgsrc = (float*) mxMalloc( sizeof(float)*M*N );
			double_to_float(imgsrc, dsrc, M*N);
			break;
		case mxUINT8_CLASS:
			isrc = (unsigned char*) mxGetPr(prhs[0]);
			imgsrc = (float*) mxMalloc( sizeof(float)*M*N );
			uint8_to_float(imgsrc, isrc, M*N);
			break;
		default:
			mexErrMsgTxt("Input image needs to be single, double, or uint8.");
			break;
	}
	
	//create outputs
	plhs[0] = mxCreateNumericMatrix(M, N, mxSINGLE_CLASS, mxREAL); 
	plhs[1] = mxCreateNumericMatrix(M, N, mxSINGLE_CLASS, mxREAL);
	localmean = (float*) mxGetPr(plhs[0]);
	localstd = (float*) mxGetPr(plhs[1]);

	localstats(localmean, localstd, imgsrc, M, N, sigma);


	//clean up
	if ( imgclass == mxDOUBLE_CLASS || imgclass == mxUINT8_CLASS )
		mxFree(imgsrc);
}