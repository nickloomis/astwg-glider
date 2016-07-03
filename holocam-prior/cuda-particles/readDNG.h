
// generic includes, mostly from hasselblad_load_v2

#include <stdlib.h> //basics + memory handling
#include <stdio.h> //file i/o
#include <time.h> //for timestamps
#include <string.h>
#include <sys/types.h>
#include <limits.h> //for int_max, etc


#include "mex.h" //the matlab mex headers





/* *************************
// 
// The following is from hasselbladShared.h
//
// ************************* */

#define CLASS

#define MAX_IFDS 10

//"aka"
typedef unsigned short ushort;
typedef unsigned char uchar;
typedef unsigned long long UINT64;


/*
   Not a full implementation of Lossless JPEG, just
   enough to decode Canon, Kodak and Adobe DNG images.	
 */
struct jhead {
  int bits, high, wide, clrs, sraw, psv, restart, vpred[6];
  ushort *huff[6], *free[4], *row;
};



typedef struct {

	struct tiff_ifd_s {
		int width, height, bps, comp, phint, offset, flip, samples, bytes;
	} tiff_ifd[MAX_IFDS]; //array of tiff_ifd's -- should only need two for 3fr's
	int tiff_nifds; //number of ifds 
	unsigned thumb_misc;
	char make[64], model[64], desc[512], cdesc[5];

	ushort raw_width, raw_height, height, width, top_margin, left_margin; //, iheight, iwidth;
	unsigned tiff_bps, tiff_compress, tiff_samples, maximum, filters;
	float iso_speed;
	off_t data_offset;
	time_t timestamp;
	int datevec[6]; //like matlab's datevec
	int tiff_flip, flip, raw, colors;

	unsigned dng_version;
	unsigned tile_width, tile_length;
	unsigned zero_after_ff; //jpeg decoding, the serious kind.

	short order; //byte order, either big or little endian.

} globals3fr; //global variables; keeping the 3fr name since it's the same data, but in a different order.


//2- and 4-byte reads, using big/little endians stored in g3 struct.
ushort CLASS sget2 (globals3fr *g3, uchar *s);
ushort CLASS get2(FILE *ifp, globals3fr *g3);
unsigned CLASS sget4 (globals3fr *g3, uchar *s);
unsigned CLASS get4(FILE *ifp, globals3fr *g3);
unsigned CLASS getint (FILE *ifp, globals3fr *g3, int type);

//#define sget4(g3, s) sget4(g3, (uchar *)s)

//simple calls that get shared
void CLASS get_timestamp (FILE *ifp, globals3fr *g3);
void CLASS parse_exif (FILE *ifp, globals3fr *g3, int base);
void CLASS parse_makernote (FILE *ifp, globals3fr *g3, int base, int uptag);
void CLASS tiff_get (FILE *ifp, globals3fr *g3, unsigned base,
	unsigned *tag, unsigned *type, unsigned *len, unsigned *save);








/* ***************************************
//
// The following is from hasselblad_load_v2.h
//
// **************************************** */


bool load3FR(const char *filename);
void identify(FILE *ifp, globals3fr *g3);
void CLASS parse_tiff(FILE *ifp, globals3fr *g3, int base); 
int CLASS parse_tiff_ifd (FILE *ifp, globals3fr *g3, int base);


void CLASS hasselblad_load_raw(FILE *ifp, globals3fr *g3, ushort (*image)[4] );
void CLASS hasselblad_load_raw_float(FILE *ifp, globals3fr *g3, float *imgPtr);
int CLASS ljpeg_start (FILE *ifp, struct jhead *jh, int info_only);
void CLASS ljpeg_end (struct jhead *jh);
ushort * CLASS make_decoder_ref (const uchar **source);
unsigned CLASS ph1_bithuff (int nbits, ushort *huff);
void reset_tiff_ifds(globals3fr *g3);
void reset_globals(globals3fr *g3);




/* ****************************************
//
// The following is from readDNG_externals.h
//
***************************************** */

void CLASS identifyDNG(globals3fr *g3, FILE *ifp);
void CLASS adobe_dng_load_raw_lj(FILE *ifp, globals3fr *g3, float *imgPtr);
void reset_globalsDNG(globals3fr *g3);



/* *****************************************
//
// The following are from readDNG.h (in HoloPod_Scientific)
//
**************************************** */


void CLASS adobe_copy_pixel (globals3fr *g3, float *imgPtr, int row, int col, ushort **rp);
int CLASS ljpeg_start_dng (FILE *ifp, globals3fr *g3, struct jhead *jh, int info_only);
void CLASS ljpeg_end_dng (struct jhead *jh);
ushort * CLASS make_decoder_ref_dng (const uchar **source);
ushort * CLASS ljpeg_row (int jrow, struct jhead *jh, FILE *ifp, globals3fr *g3);
int CLASS ljpeg_diff (FILE *ifp, globals3fr *g3, ushort *huff);
unsigned CLASS getbithuff (FILE *ifp, globals3fr *g3, int nbits, ushort *huff);
void CLASS parse_tiff_dng(int base, globals3fr *g3, FILE *ifp);
int CLASS parse_tiff_ifd_dng (FILE *ifp, globals3fr *g3, int base);
void reset_tiff_ifds_dng(globals3fr *g3);


//prototypes for utility functions
ushort CLASS sget2_dng (uchar *s);
ushort CLASS get2_dng(FILE *ifp);
unsigned CLASS sget4_dng (uchar *s);
unsigned CLASS get4_dng(FILE *ifp);
unsigned CLASS getint_dng (FILE *ifp, int type);
#define getbits(ifp,g3,n) getbithuff(ifp,g3,n,0)
#define gethuff(ifp,g3,h) getbithuff(ifp,g3,*h,h+1)





/* ********************************************
//
// The following are from sharedObjects.h
//
*********************************************** */


typedef enum {
	BMP,
	JPEG,
	TIFF,
	HASSELBLAD_RAW,
	HASSELBLAD_DNG,
	LFD,
	UNSUPPORTED
} imageTypeEnum;



//key data about images: this structure stores everything about their CPU/host-side representation.
typedef struct {
	float *ptr; //pointer to the float-cast image data
	int width, height, bpp, comp; //bits per pixel, compression key
	size_t nbytes; //number of bytes in the float-cast image
	size_t nbytesOrig, byteOffset; // number of bytes in the original image (before type-casting to float)
	imageTypeEnum type; //the type of image (determines how to work with it in the GPU)
	char *error; //describes any errors encountered while reading the file
	time_t timestamp; //TODO: implement this functionality, either reading it from the file
	//contents or from the filesystem
} imageDataStruct;





/*************************************
//
// From MyImageIO.h
//
*********************************** */

//3FR
bool read3FRHeader(FILE *ifp, imageDataStruct *image, globals3fr *g3);
bool read3FRFloat(FILE *ifp, imageDataStruct *image, globals3fr *g3);

//DNG (Hasselblad only!)
bool readDNGHeader(FILE *ifp, imageDataStruct *image, globals3fr *g3);
bool readDNGFloat(FILE *ifp, imageDataStruct *image, globals3fr *g3);

