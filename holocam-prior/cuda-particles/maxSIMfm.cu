
// TODO(nloomis): TOF docs
//
// reconstructs a hologram on a number of planes and find the magnitude and (steerable) gradient 
// at each depth. the maximum st_grad * (1 - mag) is recorded as a focus metric.
//
//
// Overview of steps in this code:
//
// load image into cpu memory (raw hologram, captured by camera); the image is
//   expected to be 2D (no RGB channels), but may be raw data from a Bayer-
//   patterned sensor
// transfer image from CPU to GPU memory
//
// take 2D FFT of the image --> fft2_of_hologram (size 2k x 2k, eg)
//
// for (z : reconstruction_depths) {
//	 compute a reconstruction kernel directly in the Fourier domain (same size
//     as fft2_of_hologram; 2k x 2k, eg)
//   multiply fft2_of_hologram * reconstruction_kernel
//   take Inverse 2D FFT of (fft2_of_hologram * reconstruction_kernel) 
//     --> reconstructed_image
//   
//   filter reconstructed_image with a gradient filter ("steerable derivative")
//     --> "gradient_magnitude_gpu" (steerable filter magnitude)
//   inverting reconstructed_image so that objects are bright and background
//     is dark -> dark_field_image
//   compute a focus metric, sim = gradient_magnitude_gpu * dark_field_image
//
//   compare the value of the focus metric for this z slice against the prior
//     best focus metric, keeping track of which z slice maximimizes the focus
//     metric
// }
//
// output from loop: map of where each pixel was best focused (same size as the
//   hologram, eg 2k x 2k); get both the depth and how in focus the pixel was
//
// transfer maps from GPU to CPU memory and back to Matlab
// 

#include "mex.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include "float.h"


// Converts a double to a float and stores the result back in the dest pointer.
// The float is downcast, discarding the least significant bits in the double.
//
// This is useful because CUDA prefers floats for its computations, while
// Matlab defaults to doubles.
void double_to_float(float* dest, double* source, const int n) {
	for (int i = 0; i < n; ++i)
		dest[i] = (float) source[i];
}


// Convert an unsigned char (or a uint8 in Matlab's type name scheme) to a
// float and store the result to dest. The char is converted to the nearest
// float. For example, 8 becomes 8.0. 
void uint8_to_float(float* dest, unsigned char* source, const int n) {
	for (int i = 0; i < n; ++i)
		dest[i] = (float) source[i];
}


// Performs integer division, but rounds up if the division is not exact.
// For example, 3 / 2 yields 1 for normal integer division, but roundUpDiv(3/2)
// returns 2.
int roundUpDiv(int num, int denom) {
	return (num / denom) + (!(num % denom) ? 0 : 1);
}


// Returns a * b, where a and b are complex-valued floats.
// This device kernel only runs on a CUDA device.
__device__ cufftComplex complexMult32(cufftComplex a, cufftComplex b) {
	cufftComplex tempResult;
	tempResult.x = a.x * b.x - a.y * b.y;
	tempResult.y = a.y * b.x + a.x * b.y;
	return tempResult;
}


// Returns a * sc, where a is a complex-valued float and sc is a scalar.
// This device kernel can only runs on a CUDA device.
__device__ cufftComplex complexScale32(cufftComplex a, float sc) {
	cufftComplex tempResult;
	tempResult.x = a.x*sc;
	tempResult.y = a.y*sc;
	return tempResult;
}


// Returns the magnitude of the complex-valued float, a.
// This device kernel only runs on a CUDA device.
__device__ float complexMagnitude(float2 a) {
	return sqrtf( a.x * a.x + a.y * a.y );
}


#include "propagation_kernels.cu" //note: included here because the file
//references complexScale32 and complexMult32... I didn't want to edit 
//propagation_kernels.cu, and I'm too lazy right now to include headers.
//note that this is currently terrible coding, you should include headers.
// -nl, 2010



// CUDA threads and data arrangement
// =========================================
// 
// CUDA launches multiple compute threads, each expected to perform
// independently of the other threads. Threads are not all launched
// simultaneously, but are staged into smaller sets of threads known as blocks.
// The user can set the size of a block to optimize performance. The user can
// also set the shape of a block (for optimization or convenience). Block shapes
// can be multidimensional.
//
// A 2D block was used here for convenience to cover all the pixels of the 
// hologram. Note that other shapes, especially those which make use of memory
// concurrency, may actually give better performance.
//
// The number of blocks is also set by the user in order to best cover the
// data they want processed. If the size of the data is not evenly divisible
// by the size of a block, there will necessarily be some threads launched in
// some blocks which do not have corresponding data. It is the CUDA thread's
// responsibility to check for whether it has data it should access.
//
// Functions marked as __global__ are passed two additional arguments: the size
// of a block, and the index of the thread within the block. The thread is
// expected to use the size and index information to determine which data
// elements it needs to access. All the kernels here use a similar pattern:
//
//   static __global__ void DeviceKernel(T* dest, U* source,
//                                       additional args...,
//                                       const int M, const int N) {
//     // Find the pixel that this thread is responsible for computing, in
//     // two dimensional (x, y) coordinates.
//	   int xidx = threadIdx.x + blockDim.x * blockIdx.x;
//     int yidx = threadIdx.y + blockDim.y * blockIdx.y;
//     // Check that the pixel is within the known bounds of the source (and
//     // destination) data. If it is NOT within the bounds, no memory access is
//     // allowed and the thread exits early; segfaults and memory leaks can
//     // occur otherwise.
//     if (xidx < N && yidx < M) {
//       int idx = yidx * N + xidx;
//       // Compute something using data from source[idx] and storing back to
//       // dest[idx].
//     }
//   }
//


// Converts a float to a complex float, storing the result into dest. The real
// part of dest is copied from source, while the imaginary part is set to zero. The
// size of the source array is M rows and N columns.
static __global__ void FloatToComplexKernel(float2* dest, float* source,
                                              const int M, const int N) {
	int xidx = threadIdx.x + blockDim.x * blockIdx.x;
	int yidx = threadIdx.y + blockDim.y * blockIdx.y;
	if (xidx < N && yidx < M){
		// Index of the data in the array.
		int idx = yidx * N + xidx;
		// dest.x is the real component.
		dest[idx].x = source[idx];
		// des.y is the imaginary component.
		dest[idx].y = 0;
	}
}


// In-place multiplication. The dest is a pointer to an array with size M rows
// and N columns; each value in dest is multipled by the constant scalar, sc.
static __global__ void scaleFloatKernel(float* dest, const float sc,
                                        const int M, const int N) {
	int xidx = threadIdx.x + blockDim.x * blockIdx.x;
	int yidx = threadIdx.y + blockDim.y * blockIdx.y;
	if (xidx < N && yidx < M){
		int idx = yidx * N + xidx;
		dest[idx] = dest[idx] * sc;
	}
}


// Copies the real part of the value in csrc and stores it to dest. Both csrc
// and dest have M rows and N columns.
static __global__ void copyRealKernel(float* dest, float2* csrc,
                                      const int M, const int N) {
	int xidx = threadIdx.x + blockDim.x * blockIdx.x;
	int yidx = threadIdx.y + blockDim.y * blockIdx.y;
	if (xidx < N && yidx < M){
		int idx = yidx * N + xidx;
		dest[idx] = csrc[idx].x;
	}
}

// Computes the magnitude of the complex-valued csrc pixel and stores it to dest.
// Both csrc and dest have M rows and N columns.
static __global__ void copyMagnitudeKernel(float* dest, float2* csrc,
                                           const int M, const int N) {
	int xidx = threadIdx.x + blockDim.x * blockIdx.x;
	int yidx = threadIdx.y + blockDim.y * blockIdx.y;
	if (xidx < N && yidx < M){
		int idx = yidx * N + xidx;
		dest[idx] = complexMagnitude(csrc[idx]);
	}
}


// TODO(nloomis): docs, with args...
//
// Compares a new floating point value against the last-known maximum to see
// if it is larger. If the new value is larger, the value is stored back as the
// new last-known maximum, and an index associated with where the maximum
// occurred is recorded to max_index.
//
// This kernel is used for finding which z-plane maximimzes a focus metric.
//
// Arguments:
//  max_metric : the current maximum value of the focus metric at each pixel;
//               if the new slice has a higher metric, the value replaces the
//               entry in max_metric.
//  index_of_max : an index recording the slice where the maximum metric
//               occurred. If the new slice's metric is larger than the
//               current maximum, the index associed with the new slice replaces
//               the entry in index_of_max.
//  slice_metric : a focus metric computed for a new reconstruction slice. If
//               slice_metric > max_metric, slice_metric replaces max_metric and
//               slice_index is stored to index_of_max.
//  slice_index : an index associated with the current slice, used to refer
//               back to the particular reconstruction.
//  M, N: max_metric, index_of_max, and slice_metric all have M rows and N
//               columns.
static __global__ void compareRealKernel(float* max_metric,
                                         short* index_of_max,
                                         float* slice_metric,
                                         const int slice_index,
                                         const int M,
                                         const int N) {
	int xidx = threadIdx.x + blockDim.x * blockIdx.x;
	int yidx = threadIdx.y + blockDim.y * blockIdx.y;
	if (xidx < N && yidx < M){
		int idx = yidx * N + xidx;
		float newreal = slice_metric[idx];
		float oldreal = max_float[idx];
		max_metric[idx] = newreal > oldreal ? newreal : oldreal;
		index_of_max[idx] = newreal > oldreal ?
			                (short) slice_index : index_of_max[idx];
	}
}


// Compares the focus metric of a reconstruction slice against the current
// maximum focus metric. If the pixel has a higher focus metric, the slice's
// focus metric, the slice's reconstruction intensity*, and the slice's
// reference index replace the values in max_metric, intensity_at_max_metric,
// and index_of_max_metric.
//
// This kernel is used to find the slice where the focus metric maximizes and
// the resulting intensity on that slice.
//
// *Reconstructions have both a real part and a complex part. The true intensity
// corresponds to the magnitude of the reconstruction. Because of the way
// propagation_kernels calculates the phase offset, the real component reflects
// most of the variation which occurs in intensity. It also requires fewer
// computations than the magnitude, and looks better to an observer. This
// function copies the real component into the intensity_at_max_metric array.
static __global__ void compareRealKernel(float* max_metric,
                                         short* index_of_max_metric,
                                         float* slice_metric,
                                         float* intensity_at_max_metric,
                                         float2* slice_intensity,
										 const int slice_index,
										 const int M,
										 const int N) {
	int xidx = threadIdx.x + blockDim.x * blockIdx.x;
	int yidx = threadIdx.y + blockDim.y * blockIdx.y;
	if (xidx < N && yidx < M){
		int idx = yidx * N + xidx;
		float newreal = slice_metric[idx];
		float oldreal = max_metric[idx];
		max_metric[idx] = (newreal > oldreal ? newreal : oldreal);
		index_of_max_metric[idx] = (newreal > oldreal ?
		                           (short) slice_index :
		                           index_of_max_metric[idx]);
		// If the focus metric is higher at this slice, record the intensity
		// (or, as a proxy, the real component in the x field) of this slice's
		// reconstruction; if not, keep the value the same.
		intensity_at_max_metric[idx] = (newreal > oldreal ?
		                               slice_intensity[idx].x :
		                               intensity_at_max_metric[idx]);
	}
}

 
// Compares the real value of a reconstruction slice against the smallest known
// real value of previous reconstructions. If the slice's value is smaller, the
// slice's value and the index of the slice overwrite the entries in the
// minimum_real and index_at_min arrays. The arrays all have M rows and N
// columns.
static __global__ void compareReal2Kernel(float* minimum_real,
                                          short* index_at_min,
                                          float2* reconstructed_slice,
										  const int slice_index,
										  const int M, const int N) {
	int xidx = threadIdx.x + blockDim.x * blockIdx.x;
	int yidx = threadIdx.y + blockDim.y * blockIdx.y;
	if (xidx < N && yidx < M){
		int idx = yidx * N + xidx;
		float slice_real_value = reconstructed_slice[idx].x;
		float current_minimum = minimum_real[idx];
		minimum_real[idx] = (slice_real_value < current_minimum ?
		                     slice_real_value :
		                     oldreal);
		index_at_min[idx] = (slice_real_value < current_minimum ?
		                    (short) slice_index :
		                    index_at_min[idx]);
	}
}


// In-place computation of the magnitude of a complex number. The arrray of
// complex numbers, source, has M rows and N columns. The magnitude of source is
// stored back to the real part of source, and the imaginary part is set to zero.
static __global__ void computeMagKernel(float2* source,
                                        const int M,
                                        const int N) {
	int xidx = threadIdx.x + blockDim.x * blockIdx.x;
	int yidx = threadIdx.y + blockDim.y * blockIdx.y;
	if (xidx < N && yidx < M) {
		int idx = yidx * N + xidx;
		source[idx].x = complexMagnitude(source[idx]);
		source[idx].y = 0;
	}
}

// Multiplies arrays of complex numbers in source1 and source2 and stores the
// result to dest. The arrays all have M rows and N columns.
static __global__ void ComplexMultiplicationKernel(float2* dest,
                                                   float2* source1,
                                                   float2* source2,
                                                   const int M,
                                                   const int N) {
	int xidx = threadIdx.x + blockDim.x * blockIdx.x;
	int yidx = threadIdx.y + blockDim.y * blockIdx.y;
	if (xidx < N && yidx < M) {
		int idx = yidx * N + xidx;
		dest[idx] = complexMult32(source1[idx], source2[idx]);
	}
}


// Returns the total magnitude of a gradient filter, computed as the L2 norm
// of the x- and y-direction gradients. The x- and y-direction gradients are
// assumed to have been computed on a complex-valued field, and the magnitudes
// of their complex responses are calculated first before finding the net
// response.
//
// The gradient magnitude is computed as:
//  x_mag = sqrt(real(x)^2 + imag(x)^2)
//  y_mag = sqrt(real(y)^2 + imag(y)^2)
//  grad_mag = sqrt(x_mag^2 + y_mag^2)
//
// gradient_magnitude: resulting magnitude of the image gradient
// gradient_in_x: the x-direction response of a gradient filter applied to the
//             complex-valued image
// gradient_in_y: the y-direction response of a gradient filter applied to the
//             complex-valued image
// M, N: number of rows and columns in each of the data arrays
//
static __global__ void GradientMagnitudeKernel(float* gradient_magnitude,
                                              float2* gradient_in_x,
                                              float2* gradient_in_y, 
	                                          const int M,
	                                          const int N) {
	int xidx = threadIdx.x + blockDim.x * blockIdx.x;
	int yidx = threadIdx.y + blockDim.y * blockIdx.y;
	if (xidx < N && yidx < M) {
		int idx = yidx * N + xidx;
		float x_magnitue = complexMagnitude(gradient_in_x[idx]);
		float y_magnitude = complexMagnitude(gradient_in_y[idx]);
		gradient_magnitude[idx] = sqrtf(x_magnitude * xmagnitude +
		                                y_magnitude * y_magnitude);
	}
}	


// SIM is the product of the gradient ("S") and image magnitude ("IM"). The
// metric considers sharp edgegs which are dark to be favorable. Since "dark" 
// has a smaller numerical value, the image brightness, or local magnitude, is
// subtracted from the expected maximum intensity so that 
//
//   (maximum_intensity - image_magnitude)
//
// gives a large value for favorable dark pixels. The SIM is then
//
//   SIM = gradient_magnitude * (maximum_intensity - image_magnitude).
//
// sim: output array of the SIM focus metric
// gradient_magnitude: magnitude of the a gradient filter which has been applied
//           to the complex-valued reconstructed image
// image_magnitude: magnitude of a single slice's reconstructed image
// maximum_intensity: maximum intensity expected over all slices; should be a
//           constant across slices
// M: number of rows in sim, gradient_magnitude, and image_magnitude
// N: number of cols in sim, gradient_magnitude, and image_magnitude
static __global__ void computeSIMkernel(float* sim, 
	                                    float* gradient_magnitude,
                                        float* image_magnitude,
                                        const float maximum_intensity, 
										const int M, const int N) {
	int xidx = threadIdx.x + blockDim.x * blockIdx.x;
	int yidx = threadIdx.y + blockDim.y * blockIdx.y;
	if (xidx < N && yidx < M) {
		int idx = yidx * N + xidx;
		sim[idx] = gradient_magnitude[idx] *
		          (maximum_intensity - image_magnitude[idx]);
	}
}


// In-place addition of shorts. The kernel is used to add value to all elements
// of dest. The dest array has M rows and N columns.
static __global__ void AddShortsKernel(short* dest, const short value,
                                      const int M, const int N) {
	int xidx = threadIdx.x + blockDim.x * blockIdx.x;
	int yidx = threadIdx.y + blockDim.y * blockIdx.y;
	if (xidx < N && yidx < M){
		dest[yidx * N + xidx] += value;
	}
}


/******************************************************************************
FFT coordinates

CUDA's FFT follows the same pattern as the popular FFTW library and arranges its
output data in a computationally-efficient but confusing pattern. The first half
of the data corresponds to the positive frequency components, while the last
half of the data corresponds to the negative frequency components. CUDA also
makes use of the fact that the component at the positive Nyquist frequency is
the same as at the negative Nyquist frequency.

The output is arranged as:

  [0, 1, 2, 3, ..., N/2 - 1, N/2, -N/2+1, -N/2+2, ... -1]

  |<------------------------>fn<----------------------->|
       positive frequency          negative frequency
           components                 components

where the N/2 sample is the Nyquist frequency component and is the same for both
the positive and negative Nyquist frequency.

Functions which compute values directly in the Fourier domain first find the
frequency component for each array entry.
******************************************************************************/


// Applies a specialized low-pass filter to data which has been 2D Fourier
// transformed.
//
// A "power" filter is a low-pass filter with a controllable ramp down, and
// follows
//
//   P(u, v) = exp(-(u^p + v^p) / sigma)
//
// where p is a positive even integer, sigma is a scale parameter, and u and
// v are normalized frequences. Normalization is against the Nyquist frequency,
// so that u and v range from [-1, 1]. Setting p = 2 gives a Gaussian low-pass
// filter; the higher p, the higher the cut-off frequency becomes and the
// sharper the ramp down.
//
// The sigma parameter can be used to set the scale, and functions like an
// additive offset for high frequencies (so that a small, controllable amount of
// the higher spatial frequencies can be kept). At |u| = |v| = 1, the filter has
// a value of P = exp(-1 / sigma).
//
// The power filter is applied directly to the 2D FFT data in fsrc, multiplying
// the magnitude of the component. The power_order is p, and should be an even
// integer. M is the number of rows and N is the number of columns in fsrc.
// The data is modified in-place.
static __global__ void PowerKernel(float2* fsrc,
                                   const int N, const int M,
								   const float power_order,
								   const float sigma) {
	int xidx = threadIdx.x + blockDim.x * blockIdx.x;
	int yidx = threadIdx.y + blockDim.y * blockIdx.y;
	if (xidx < N && yidx < M){
		int idx = yidx * N + xidx;

		//get the FFTW-shifted u,v coordinates (in numbers of samples)
		//option 4) use normalized freqs (just the magnitude) - note that uidx->u here in the nomenclature.
        // Find the sample corresponding to the Nyquist frequency.
		int nyquist_sample_x = N / 2;
		int nyquist_sample_y = M / 2;

        // Compute the normalized frequency for each pixel.
        // Each pixel's corresponding Fourier frequency is found using the
        // shifted coordinates discussed above, then normalized so that the
        // Nyquist frequency is 1.
		float normalized_freq_u = (xidx >= nyquist_sample_x ?
            (float) (N - xidx) / (float) nyquist_sample_x :
		    (float) xidx / (float) nyquist_sample_x);
		float normalized_freq_v = (yidx >= nyquist_sample_y ?
            (float) (M - yidx) / (float) nyquist_sample_y :
		    (float)yidx / (float) nyquist_sample_y);

        // Pull out the coefficient for this pixel's frequency.
		float2 value = fsrc[idx];

        // Compute the power kernel's magnitude.
		float fkeep = __expf(-( __powf(normalized_freq_u, power_order) +
		                        __powf(nonrmalized_freq_v, power_order)) /
		                     sigma);

        // Scale the FFT coefficient's magnitude and store it back to its
        // original array.
		value.x = value.x * fkeep;
		value.y = value.y * fkeep; 
		fsrc[idx] = value;
	}
}

// Computes the convolution kernel for a steerable filter in the x-direction. A
// steerable filter has two important characteristics:
//  1) the filter includes some smoothing, so the effects of random noise are
//     reduced, and
//  2) the gradient magnitude is independent of the orientation of an edge in
//     an image.
// 
// TODO(nloomis): finish docs.
// note: use float2 so that fft2 can be taken immediately... or is it in freq
// domain already??
static __global__ void ComputeSteerableFilterXCoefsKernel(float2* gradient_filter_in_x,
                                        const float sigma, 
                                        const int N, const int M) {
	int xidx = threadIdx.x + blockDim.x * blockIdx.x;
	int yidx = threadIdx.y + blockDim.y * blockIdx.y;
	if (xidx < N && yidx < M){
        int idx = yidx * N + xidx;
        
		int halfx = N / 2;
		int halfy = M / 2;
		float u = (xidx >= halfx ? -(float)(N - xidx) : (float) xidx);
		float v = (yidx >= halfy ? -(float)(M - yidx) : (float) yidx);
		
		//set the value
		gradient_filter_in_x[idx].x = -2.0f * u * __expf(-(u * u + v * v) /
		                             (2.0f * sigma * sigma));
        // The imaginary component is zero.
        gradient_filter_in_x[idx].y = 0;

	}
}


// Computes the convolution kernel for a steerable filter in the y-direction.
// See documentation for ComputeSteerableFilterXCoefsKernel above.
//
static __global__ void ComputeSteerableFilterYCoefsKernel(float2* gradient_filter_in_y, const float sigma, 
                                        const int xpix, const int ypix) {
	int xidx = threadIdx.x + blockDim.x * blockIdx.x;
	int yidx = threadIdx.y + blockDim.y * blockIdx.y;
	if (xidx < xpix && yidx < ypix){
		int halfx = xpix / 2;
		int halfy = ypix / 2;
		float u = (xidx >= halfx ? -(float)(xpix - xidx) : (float)xidx);
		float v = (yidx >= halfy ? -(float)(ypix - yidx) : (float)yidx);

		int idx = yidx * ypix + xidx; //the linear index of the data
		
		//set the value
		gradient_filter_in_y[idx].x = -2.0f * v * __expf(-(u * u + v * v) /
			            (2.0f * sigma * sigma));
        // The imaginary component is zero.
        gradient_filter_in_y[idx].y = 0;
	}
}

// Hint: it would have been much better to create structs to hold the parameters
// and options, reducing the number of variables passed in.
// Hint: the mode flags would have been better as enums than as bools.

void Reconstruct(float* focus_metric,
                 short* index_of_max_focus_metric,
                 float* min_intensity_cpu,
                 short* index_of_min_intensity,
                 float* hologram_cpu, 
			     double* reconstruction_z_depths,
                 const int num_z_depths,
                 const int M,
                 const int N, 
				 const double wavelength,
                 const double pixel_size,
				 const float power_filter_order,
                 const float power_filter_offset,
				 const float steerable_filter_sigma,
                 const float maximum_intensity,
				 float* intensity_at_best_focus_cpu,
                 bool record_min_intensity,
                 bool record_intensity_at_best_focus) {
	float* hologram_gpu;
    float* min_intensity_gpu;
    float* max_sim_focus_metric_gpu;
    float* gradient_magnitude_gpu;
    float* image_magnitude;
    float* sim;
    float* intensity_at_best_focus_gpu;
	short* index_of_min_intensity_gpu;
    short* index_of_max_focus_metric_gpu;
	float2* complex_hologram_gpu;
    float2* reconstruction_gpu;
    float2* gradient_filter_in_x;
    float2* gradient_filter_in_y;
    float2* gradient_response_in_x_gpu;
    float2* gradient_response_in_y_gpu;
	float adudu, cutoff2, fftscale;
	float2 offsetPhaseExp;
	float sigma;
    //...for both x and y dims in the block
    const int kBlockSize = 16;

    // A block is the number of threads which are launched at each round. A size
    // of 16x16 is used to step through the image 256 threads at a time. For
    // CUDA (as of 2010), threads are launched internally in multiples of 16,
    // so I used a multiple of 16 here for better processing. The exact size and
    // shape were not optimized.
	dim3 myblock(kBlockSize, kBlockSize);

    // A grid describes to the GPU how many times it should launch thread
    // blocks of a given size. There are enough grids so that one thread is
    // launched for every pixel. If N or M are not evenly divisible by the block
    // size, there will be more threads launched than pixels (due to the limited
    // granularity the GPU gives in its thread blocks), so the kernels all
    // include tests to make sure its thread corresponds to a pixel.
	dim3 mygrid( roundUpDiv(N, kBlockSize), roundUpDiv(M, kBlockSize) );

	// Upload the hologram to the GPU's memory.
    mexPrintf("Uploading hologram\n");
	cudaMalloc((void**) &hologram_gpu, sizeof(float) * M * N);
    cudaMemcpy(hologram_gpu,
               hologram_cpu,
               sizeof(float) * M * N,
               cudaMemcpyHostToDevice);

    // Convert the hologram from real-valued floats to complex-valued floats.
    // This is needed because we need the complex-valued FFT for computing the
    // physical wave propagation.
	cudaMalloc((void**) &complex_hologram_gpu, sizeof(float2) * M * N);
	FloatToComplexKernel<<<mygrid, myblock>>>(complex_hologram_gpu,
                                              hologram_gpu,
                                              M, N);
    // After the FFT is computed, the original source data does not need to be
    // maintained on the GPU.
	cudaFree(hologram_gpu);

	// Allocate memory for the remaining variables in the GPU.
    mexPrintf("Creating memory on GPU\n");
	cudaMalloc((void**) &max_sim_focus_metric_gpu, sizeof(float) * M * N);
	cudaMalloc((void**) &index_of_max_focus_metric_gpu, sizeof(short) * M * N);
	cudaMemset(index_of_max_focus_metric_gpu, 0, sizeof(short) * M * N);
	if (record_min_intensity){
		cudaMalloc((void**) &min_intensity_gpu, sizeof(float) * M * N);
		cudaMalloc((void**) &index_of_min_intensity_gpu, sizeof(short) * M * N);
		cudaMemset(index_of_min_intensity_gpu, 0, sizeof(short) * M * N);
	}

	// Create an FFT plan. The plan tells the GPU how to compute a Fourier
    // transform, its dimensionality, and the complexity. There is a TINY
    // amount of overhead while CUDA "plans" (optimizes) the computation, so
    // that it will work extremely fast later. The plan is meant to be re-used.
    mexPrintf("Creating FFT plan\n");
	cufftHandle plan;
    // CUFFT_C2C indicates that the transform is from complex-valued data and
    // that the results should be stored as complex-valued.
	cufftPlan2d(&plan, N, M, CUFFT_C2C);

    // Compute the 2D FFT of the hologram data, in place, yielding the spatial
    // frequencies of the image.
	mexPrintf("Executing FFT\n");
    // CUFFT_FORWARD indicates that this transform is a forward Fourier
    // transform (not an inverese FFT).
	cufftExecC2C(plan,
                 complex_hologram_gpu,
                 complex_hologram_gpu,
                 CUFFT_FORWARD);

// TODO(nloomis): explain sigma on power filter; finish docs for power filter

    // Holograms may be captured using a monochrome sensor or a sensor with a
    // Bayer filter. In the Bayer'ed holograms, the effect of the color filter
    // manifests itself as replicated spectra at the Nyquist frequencies. A
    // power filter acts as a low-pass filter with a smooth transition at a
    // controllable rate and location in the frequency domain.
	mexPrintf("Applying power filter\n");
    // Cheap trick: if power_filter_order is negative, the filter isn't applied.
	if (power_filter_order > 0) {
		// mexPrintf("Applying power kernel\n");
		sigma = -1.0f / log(power_filter_offset);
		PowerKernel<<<mygrid, myblock>>>(complex_hologram_gpu, N, M,
			power_filter_order, power_filter_offset, sigma);
	}

    // Compute the steerable filter, a gradient filter with smoothing. The
    // gradient filter is applied to each reconstructed slice.
    mexPrintf("Computing steerable filter\n");
	cudaMalloc((void**) &gradient_filter_in_x, sizeof(float2) * M * N);
	cudaMalloc((void**) &gradient_filter_in_y, sizeof(float2) * M * N);
	ComputeSteerableFilterXCoefsKernel<<<mygrid, myblock>>>(
        gradient_filter_in_x,
        steerable_filter_sigma,
        M, N);
	ComputeSteerableFilterYCoefsKernel<<<mygrid, myblock>>>(
        gradient_filter_in_y,
        steerable_filter_sigma,
        M, N);

    // Compute the filter's frequency-domain response. This makes it easy to
    // apply the filter later: instead of a slow spatial convolution, we can do
    // a point-wise multiplication in the Fourier domain and take the inverse
    // 2D FFT.
    // The forward 2D FFTs are performed in-place.
	cufftExecC2C(plan,
                 gradient_filter_in_x,
                 gradient_filter_in_x,
                 CUFFT_FORWARD);
	cufftExecC2C(plan,
                 gradient_filter_in_y,
                 gradient_filter_in_y,
                 CUFFT_FORWARD);
    // Reserve memory for the gradient response in each direction and the 
    // magnitude of the gradient response.
	cudaMalloc((void**) &gradient_response_in_x_gpu, sizeof(float2) * M * N);
	cudaMalloc((void**) &gradient_response_in_y_gpu, sizeof(float2) * M * N);
	cudaMalloc((void**) &gradient_magnitude_gpu, sizeof(float) * M * N);


	// Reserve memory for a single reconstructed slice of the volume.
	cudaMalloc( (void**)&reconstruction_gpu, sizeof(float2) * M * N);


	//reserve memory for the focus metric components
	cudaMalloc( (void**)&image_magnitude, sizeof(float) * M * N);
	cudaMalloc( (void**)&sim, sizeof(float) * M * N);
	if (record_intensity_at_best_focus)
		cudaMalloc( (void**)&intensity_at_best_focus_gpu, sizeof(float) * M * N);

	mexPrintf("Computing reconstructions\n");

	//reconstruct the hologram at each depth in z, compare it to the previous
	//minimum intensity
	for (int slice_index = 0; slice_index < num_z_depths; ++slice_index){
		//copy the fft spectrum to a variable that's temporary for each plane
		cudaMemcpy(reconstruction_gpu, complex_hologram_gpu, sizeof(float2)*M*N, cudaMemcpyDeviceToDevice);

		//set the propagation parameters
		setPropagationParams( (float)(reconstruction_z_depths[slice_index]), (float)wavelength, (float)pixel_size, M, N, 
			&adudu, &cutoff2, &offsetPhaseExp, &fftscale, false);

		//apply the propagation kernel (full... could speed this up by ignoring cutoff)
		fresnelKernel<<<mygrid, myblock>>>(reconstruction_gpu, M, N, adudu, 
			cutoff2, offsetPhaseExp, fftscale);

		//take the inverse transform: get an image at the plane.
		cufftExecC2C(plan, reconstruction_gpu, reconstruction_gpu, CUFFT_INVERSE);

		//compute the magnitude
		computeMagKernel<<<mygrid, myblock>>>(reconstruction_gpu, M, N);

		//compute the steerable gradient responses in the x and y directions
		cudaMemcpy(gradient_response_in_x_gpu, reconstruction_gpu, sizeof(float2) * M * N, cudaMemcpyDeviceToDevice); //copy the magnitude info
		cufftExecC2C(plan, gradient_response_in_x_gpu, gradient_response_in_x_gpu, CUFFT_FORWARD); //the spectrum of the magnitude
		cudaMemcpy(gradient_response_in_y_gpu, gradient_response_in_x_gpu, sizeof(float2) * M * N, cudaMemcpyDeviceToDevice); //copy the spectrum information
		//apply the sfilt kernels in the x and y directions
		ComplexMultiplicationKernel<<<mygrid, myblock>>>(gradient_response_in_x_gpu, gradient_response_in_x_gpu, gradient_filter_in_x, M, N);
		ComplexMultiplicationKernel<<<mygrid, myblock>>>(gradient_response_in_y_gpu, gradient_response_in_y_gpu, gradient_filter_in_y, M, N);
		cufftExecC2C(plan, gradient_response_in_x_gpu, gradient_response_in_x_gpu, CUFFT_INVERSE);
		cufftExecC2C(plan, gradient_response_in_y_gpu, gradient_response_in_y_gpu, CUFFT_INVERSE);
		//finally: compute the magnitude and orientation
		GradientMagnitudeKernel<<<mygrid,myblock>>>(gradient_magnitude_gpu, gradient_response_in_x_gpu, gradient_response_in_y_gpu, M, N);
		scaleFloatKernel<<<mygrid,myblock>>>(gradient_magnitude_gpu, 1.0f / ((float) (M * M) * (float) (N * N)), M, N);

		//find the local min intensity (for focus_metric)
		//localMinKernel<<<mygrid, myblock>>>(image_magnitude, reconstruction_gpu, M, N);
		copyRealKernel<<<mygrid, myblock>>>(image_magnitude, reconstruction_gpu, M, N);


		//compute the SIM metric
		computeSIMkernel<<<mygrid, myblock>>>(sim, gradient_magnitude_gpu, image_magnitude, maximum_intensity, M, N);
		

		if (record_min_intensity){
			//compare the current intensity to the previous minimum
			if (slice_index == 0)
                // If this is the first slice to be reconstructed, copy the
                // reconstruction values as initial values.
				copyRealKernel<<<mygrid, myblock>>>(min_intensity_gpu,
                                                    reconstruction_gpu,
                                                    M, N);
			else
                // Compare this reconstructed slice's intensity against the
                // previously recorded minimum intensity; pixels which have
                // lower intensities for this slice replace the values in
                // min_intensity_gpu and index_of_min_intensity_gpu. 
				compareReal2Kernel<<<mygrid, myblock>>>
                   (min_intensity_gpu,
                    index_of_min_intensity_gpu, 
                    reconstruction_gpu, 
                    slice_index, 
                    M, N);
		}

		//compare the current SIM to the previous SIMs
		if (slice_index == 0) {
			cudaMemcpy(max_sim_focus_metric_gpu, sim, sizeof(float)*M*N, cudaMemcpyDeviceToDevice);
			if (record_intensity_at_best_focus)
				cudaMemcpy(intensity_at_best_focus_gpu, min_intensity_gpu, sizeof(float) * M * N, cudaMemcpyDeviceToDevice);
		}
		else {
			if (!record_intensity_at_best_focus)
				compareRealKernel<<<mygrid, myblock>>>(
                    max_sim_focus_metric_gpu, 
                    index_of_max_focus_metric_gpu,
                    sim,
                    slice_index, 
                    M, N);
			else //record_intensity_at_best_focus == true
				compareRealKernel<<<mygrid, myblock>>>(
                    max_sim_focus_metric_gpu,
                    index_of_max_focus_metric_gpu, 
                    sim, 
                    intensity_at_best_focus_gpu, 
                    reconstruction_gpu, 
                    slice_index, 
                    M, N);
		}
	}

	// C-style indexing uses 0 as its first element, while Matlab uses 1 for its
    // first element. Add a 1 to all the slice indices recorded so that the
    // returned indices are directly usable in Matlab.
	if (record_min_intensity)
		AddShortsKernel<<<mygrid, myblock>>>(index_of_min_intensity_gpu, 1, M, N);
	AddShortsKernel<<<mygrid, myblock>>>(
        index_of_max_focus_metric_gpu, 1, M, N);

    // Copy results from the GPU back to the CPU memory.
	mexPrintf("Copying results back to host\n");
	cudaMemcpy(focus_metric, max_sim_focus_metric_gpu, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(index_of_max_focus_metric, index_of_max_focus_metric_gpu, sizeof(short) * M * N, cudaMemcpyDeviceToHost);
	if (record_min_intensity) {
		cudaMemcpy(min_intensity_cpu, min_intensity_gpu, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
		cudaMemcpy(index_of_min_intensity, index_of_min_intensity_gpu, sizeof(short) * M * N, cudaMemcpyDeviceToHost);
	}
	if (record_intensity_at_best_focus)
		cudaMemcpy(intensity_at_best_focus_cpu, intensity_at_best_focus_gpu, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

	mexPrintf("Cleaning up memory and plans\n");

	//clean up!
	cufftDestroy(plan);
	cudaFree(complex_hologram_gpu);
	cudaFree(reconstruction_gpu);
	if (record_min_intensity){
		cudaFree(min_intensity_gpu);
		cudaFree(index_of_min_intensity_gpu);
	}
	cudaFree(gradient_filter_in_x);
	cudaFree(gradient_filter_in_y);
	cudaFree(gradient_response_in_x_gpu);
	cudaFree(gradient_response_in_y_gpu);
	cudaFree(gradient_magnitude_gpu);
	cudaFree(sim);
	cudaFree(image_magnitude);
	cudaFree(index_of_max_focus_metric_gpu);
	cudaFree(max_sim_focus_metric_gpu);
	if (record_intensity_at_best_focus)
		cudaFree(intensity_at_best_focus_gpu);
}



// [focus_metric, index_of_max_focus_metric, min_intensity_cpu, index_of_min_intensity, R_at_SIM] = maxSIMfm(img, z, wavelength, pixel_size, ...
//           powerKernOrder, powerKernOffset, sigma, maximum_intensity);
void mexFunction( int nlhs, mxArray* plhs[],
                  int nrhs, const mxArray* prhs[]) {

    // Pointer for the hologram data.
	float* hologram_cpu;
    // Minimum reconstruction intensity.
    float* min_intensity_cpu;
    // Stores a metric for how well each hologram pixel can be focused.
    float* sim_focus_metric;
    // Stores the reconstructed image's intensity at the best focus.
    float* intensity_at_best_focus_cpu = NULL;
    // Pointer for source hologram data, if it is passed in as a double.
	double* dsrc;
    // Pointer for source hologram data, if it is passed in as an unsigned char.
	unsigned char* isrc;
    // Index of the slice which has the minimum intensity at that pixel; array
    // the same size as hologram_cpu.
	short* index_of_min_intensity;
    // Index of the slice which has the maximum focus metric at that pixel;
    // array the same size as hologram_cpu.
    short* index_of_max_focus_metric;
    // Array of depths to reconstruct, in meters.
	double* reconstruction_z_depths;
    // Wavelength of the illumination light in meters.
    double wavelength = 658e-9;
    // Size of a camera pixel in meters.
    double pixel_size = 9e-6;
    // Number of rows in hologram_cpu and any other array which has the same
    // size (hint: most of the arrays have the same size).
	int M;
    // Number of columns in hologram_cpu and any other array which has the
    // same size.
    int N;
    // Number of depths to reconstruct; size of z.
    int num_z_depths;
	float power_filter_order = -1;
    float power_filter_offset = .001;
    float sigma = 2;
    float maximum_intensity = 1000;
	bool record_min_intensity;
    // If true, the reconstruction intensity is recorded at each pixel for the
    // slice that best focuses that pixel.
    bool record_intensity_at_best_focus;

	if (nrhs<2)
		mexErrMsgTxt("maxSIMfm(image, z, [wavelength], [pixel_size]) requires at least two arguments.");

	//get the size of the hologram
	M = mxGetN(prhs[0]); //number of y-direction pixels
	N = mxGetM(prhs[0]); //number of x-direction pixels
	//NB: I'm mixing M, N so that they align with the Matlab imagesc plots.
	//M is the y-direction , N is the x-direction.
	//this is because Matlab stores its data reading down the rows (row-major), while
	//C is column-wise ordering (column-major)
	
	//retrieve the depths at which to do reconstructions
	reconstruction_z_depths = mxGetPr(prhs[1]);
	num_z_depths = mxGetM(prhs[1]) * mxGetN(prhs[1]);
	
	//mexPrintf("M: %i, N: %i, num_z_depths: %i\n", M, N, num_z_depths);

	//get optional parameters; use defaults otherwise
	if (nrhs>=3)
		if (!mxIsEmpty(prhs[2]))
			wavelength = mxGetScalar(prhs[2]);
	if (nrhs>=4)
		if (!mxIsEmpty(prhs[3]))
			pixel_size = mxGetScalar(prhs[3]);
	if (nrhs>=5)
		if (!mxIsEmpty(prhs[4]))
			power_filter_order = (float) mxGetScalar(prhs[4]);
	if (nrhs>=6)
		if (!mxIsEmpty(prhs[5]))
			power_filter_offset = (float) mxGetScalar(prhs[5]);
	if (nrhs>=7)
		if (!mxIsEmpty(prhs[6]))
			sigma = (float) mxGetScalar(prhs[6]);
	if (nrhs>=8)
		if (!mxIsEmpty(prhs[7]))
			maximum_intensity = (float) mxGetScalar(prhs[7]);
			
    // Retrieve the pointer to the hologram data. If the data type is not float,
    // it needs to be converted for to float for efficient computations on the
    // GPU.
	mxClassID imgclass = mxGetClassID(prhs[0]);
	switch (imgclass) {
		case mxSINGLE_CLASS:
		    // If the hologram was passed in as a float, copy its pointer.
			hologram_cpu = (float*) mxGetPr(prhs[0]);
			break;
		case mxDOUBLE_CLASS:
		    // If the hologram was passed in as a double, it needs to be
		    // down-cast to float.
			dsrc = mxGetPr(prhs[0]);
			hologram_cpu = (float*) mxMalloc(sizeof(float) * M * N);
			double_to_float(hologram_cpu, dsrc, M * N);
			break;
		case mxUINT8_CLASS:
		    // If the hologram was passsed in as an unsigned char (what Matlab
		    // calls "uint8"), it needs to be converted to float.
			isrc = (unsigned char*) mxGetPr(prhs[0]);
			hologram_cpu = (float*) mxMalloc(sizeof(float) * M * N);
			uint8_to_float(hologram_cpu, isrc, M * N);
			break;
		default:
			mexErrMsgTxt("Input image needs to be single, double, or uint8.");
			break;
	}
	
	//create outputs
	
	if (nlhs > 2)
		record_min_intensity = true;
	else
		record_min_intensity = false;
	if (nlhs > 4)
		record_intensity_at_best_focus = true;
	else
		record_intensity_at_best_focus = false;

	plhs[0] = mxCreateNumericMatrix(N, M, mxSINGLE_CLASS, mxREAL); //the order is reversed here
	plhs[1] = mxCreateNumericMatrix(N, M, mxINT16_CLASS, mxREAL); //from what you'd expect... row-major vs column-major.
	focus_metric = (float*) mxGetPr(plhs[0]);
	index_of_max_focus_metric = (short*) mxGetPr(plhs[1]);

	if (record_min_intensity) {
		plhs[2] = mxCreateNumericMatrix(N, M, mxSINGLE_CLASS, mxREAL);
		plhs[3] = mxCreateNumericMatrix(N, M, mxINT16_CLASS, mxREAL);
		min_intensity_cpu = (float*) mxGetPr(plhs[2]); //single
		index_of_min_intensity = (short*) mxGetPr(plhs[3]); //int16
	}

	if (record_intensity_at_best_focus){
		plhs[4] = mxCreateNumericMatrix(N, M, mxSINGLE_CLASS, mxREAL);
		intensity_at_best_focus_cpu  = (float*) mxGetPr(plhs[4]);
	}

	//reconstruct hologram and find mins
	Reconstruct(sim_focus_metric, index_of_max_focus_metric,
                min_intensity_cpu, index_of_min_intensity,
                hologram_cpu,
                reconstruction_z_depths, num_z_depths,
                M, N,
                wavelength, pixel_size, 
		        power_filter_order, power_filter_offset,
                sigma,
                maximum_intensity, intensity_at_best_focus_cpu,
                record_min_intensity,
                record_intensity_at_best_focus);

	// If the hologram was originally passed in as a double or unsigned chars,
	// hologram_cpu would have been malloc'd. Free it now, before exiting.
	if ( imgclass == mxDOUBLE_CLASS || imgclass == mxUINT8_CLASS )
		mxFree(hologram_cpu);
}
