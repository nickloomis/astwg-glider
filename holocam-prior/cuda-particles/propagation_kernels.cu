
#include <math.h>


//easier version of this, since there's several complex mults together.
__device__ float2 fresnelKernMult(float2 src, float cospart, float sinpart, 
										float2 phaseFactor, float scale)
{
	float2 tempValue, kernValue;
	kernValue.x = cospart;
	kernValue.y = sinpart;
	tempValue = complexScale32( complexMult32(src, complexMult32(kernValue, phaseFactor)), scale);
	return tempValue;
}



static __global__ void fresnelKernel(float2* dest, 
									  const int mpix, const int npix, float adudu, const float cutoffIdx2,
									  float2 phaseFactor, float scale)
{
	//find out where we are in the matrix
	int xidx = blockIdx.x * blockDim.x + threadIdx.x;
	int yidx = blockIdx.y * blockDim.y + threadIdx.y; 

	//decide if the thread will have data to work with, and go!
	if (xidx<npix && yidx<mpix){

		int idx = yidx*npix+xidx; 

		int uidx=npix/2-abs(npix/2-xidx); //I'm convinced there's a better way to do this
		int vidx=mpix/2-abs(mpix/2-yidx);
		long freqidx2 = __mul24(uidx,uidx) + __mul24(vidx,vidx); //squared frequency index
		//long freqidx2 = uidx*uidx + vidx*vidx;

		float thisfreq= -1*((float)freqidx2)*adudu;

		if (freqidx2 <= cutoffIdx2) //use the squared frequency cutoff to shortcut taking another sqrt
		{
			//we're inside the frequency cutoff
			float cospart, sinpart;
			__sincosf(thisfreq, &sinpart, &cospart);
			dest[idx] = fresnelKernMult(dest[idx],cospart,sinpart,phaseFactor,scale);
		}
		else {
			//we're outside the cutoff, window the damn thing
			dest[idx].x = 0;
			dest[idx].y = 0;
		} //consider writing this as a conditional ( ? : ) instead?
		
	}
}




/////////// CPU Functions ////////////



//take in angle and return exp(i*angle)
float2 cpuComplexExponential(float angle)
{
	float2 tempValue;
	tempValue.x = cosf(angle);
	tempValue.y = sinf(angle);
	return tempValue;
}


void setPropagationParams(float zprop, float lambda, float delta, int M, int N, 
						  float *adudu, float *cutoff2, float2 *offsetPhaseExp, float *fftscale,
						  bool verboseFlag)
{
	const float pi = 3.141592653589793; //a reasonable approx

    //calculate a bunch of optical properties for the kernel
	float du = 1.0f/(M*delta); //a step in frequency domain; assumes the same step in M, N directions
	float evCutoffFreq = 1/lambda; //evanescent cutoff frequency
	float nyqCutoffFreq = M*delta/(2*lambda*fabs(zprop)); //Nyquist cutoff frequency
	float limitingFreq = (evCutoffFreq<nyqCutoffFreq ? evCutoffFreq : nyqCutoffFreq); //the most limiting frequency
	float cutoffIdx = limitingFreq/du;
	
	//compute & save key variables
	*fftscale = 1.0f/(M*N); //how much to "de-scale" the FFT
	*adudu=lambda*pi*zprop*du*du; //for kernel freq, multiply this by idx*idx.
	*offsetPhaseExp = cpuComplexExponential(2*pi*zprop/lambda);
	*cutoff2 = cutoffIdx*cutoffIdx; //squared cutoff index (for uidx*uidx comparison in kernel)

    //report settings, if desired
    if (verboseFlag){
        mexPrintf("Array size:  (%i x %i)\n", M, N);
        mexPrintf("Nyquist:     %f\n", nyqCutoffFreq);
        mexPrintf("Evanescent:  %f\n", evCutoffFreq);
        mexPrintf("Limiting:    %f\n", limitingFreq);
        mexPrintf("  CutoffIdx: %i\n", (int) cutoffIdx);
        //mexPrintf("Phase factor = (%f, %fI)\n", offsetPhaseExp.x, offsetPhaseExp.y);
        mexPrintf("FFT Scaling: %e\n", *fftscale);
    }
}
