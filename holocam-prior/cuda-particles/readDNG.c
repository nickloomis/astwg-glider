//function to read in a DNG or 3FR (RAW data file from a Hasselblad or 
//Adobe Digital Negative from a Hasselblad back)
//
//call using
// img = readDNG(filename);
//



#include "readDNG.h" //header that contains *everything*




// the FC, BAYER macros

#define FC(row,col) \
	(g3->filters >> ((((row) << 1 & 14) + ((col) & 1)) << 1) & 3)
//14 = b1110
//3 = b0011
//1 = b0001

//#define BAYER(row,col) \
//	image[((row) >> shrink)*iwidth + ((col) >> shrink)][FC(row,col)]
#define BAYER(row,col) \
	image[(row) * g3->width + (col)][FC(row,col)]








// short macro-like functions for reading in data 

/* 2- and 4-byte gets which respect big- and little-endian byte storage.
// these are defined in hasselblad_load_v2, but may be used in 
// any of the dcraw() code rips.
//
*/

ushort CLASS sget2 (globals3fr *g3, uchar *s)
{
  if (g3->order == 0x4949)		/* "II" means little-endian */
    return s[0] | s[1] << 8;
  else				/* "MM" means big-endian */
    return s[0] << 8 | s[1];
}


ushort CLASS get2(FILE *ifp, globals3fr *g3)
{
  uchar str[2] = { 0xff,0xff };
  fread (str, 1, 2, ifp);
  return sget2(g3, str);
}

unsigned CLASS sget4 (globals3fr *g3, uchar *s)
{
  if (g3->order == 0x4949) //little endian
    return s[0] | s[1] << 8 | s[2] << 16 | s[3] << 24;
  else //big endian
    return s[0] << 24 | s[1] << 16 | s[2] << 8 | s[3];
}
//#define sget4(g3, s) sget4(g3, (uchar *)s)

unsigned CLASS get4(FILE *ifp, globals3fr *g3)
{
  uchar str[4] = { 0xff,0xff,0xff,0xff };
  fread (str, 1, 4, ifp);
  return sget4(g3, str);
}

unsigned CLASS getint (FILE *ifp, globals3fr *g3, int type)
{
  return type == 3 ? get2(ifp, g3) : get4(ifp, g3);
}





// macro-functions for big-endian DNG information
//
// NB: these all assume BIG-ENDIAN!! (the 3fr uses little endian, strangely.)
ushort CLASS sget2_dng (uchar *s) {
	return s[0] << 8 | s[1];
}

ushort CLASS get2_dng(FILE *ifp) {
	uchar str[2] = { 0xff,0xff };
	fread (str, 1, 2, ifp);
	return sget2_dng(str);
}

unsigned CLASS sget4_dng (uchar *s) {
	return s[0] << 24 | s[1] << 16 | s[2] << 8 | s[3];
}
#define sget4_dng(s) sget4_dng((uchar *)s)

unsigned CLASS get4_dng(FILE *ifp) {
	uchar str[4] = { 0xff,0xff,0xff,0xff };
	fread (str, 1, 4, ifp);
	return sget4_dng(str);
}


unsigned CLASS getint_dng (FILE *ifp, int type) {
  return type == 3 ? get2_dng(ifp) : get4_dng(ifp);
}






// the main mex function

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
	
	//char *filename;
	char filename[255], ext[4];
	FILE *ifp; //input file pointer
	globals3fr g3;
	imageDataStruct imagedata;
	mwSize dims[2];
	int flen;

	//for debugging:
	//strcpy(filename, "A1002233.dng"); //for now, a fixed test image
	//strcpy(filename, "A0000257.3FR");

    //parse inputs from Matlab
	if (!mxIsClass(prhs[0],"char"))
		mexErrMsgTxt("The first argument of readDNG() is the filename string.");
	flen = max(mxGetM(prhs[0]), mxGetN(prhs[0])); //number of characters in the filename array
	mxGetString(prhs[0], filename, flen+1); //note: mxGetString(src, dest, numel)!
	//mexPrintf("File: %s (%i chars)\n", filename, flen);


	//decide whether this is a 3FR or DNG
	strncpy(ext, &(filename[strlen(filename)-3]), 3);
	ext[3] = 0; //null termination
	//mexPrintf("extension: [%s]\n", ext);

	if (!strncmp(ext, "3fr", 3) || !strncmp(ext, "3FR", 3))
		imagedata.type = HASSELBLAD_RAW;
	else if ( !strncmp(ext, "dng", 3) || !strncmp(ext, "DNG", 3))
		imagedata.type = HASSELBLAD_DNG;
	else
		mexErrMsgTxt("File extension not recognized as a Hasselblad DNG or 3FR");


	//open a pointer to the file
	ifp = fopen(filename, "rb"); //open for reading
	if (ifp == NULL) {
		mexErrMsgTxt("File could not be opened for reading.");
		return;
	}

	switch (imagedata.type) {
		case HASSELBLAD_RAW:
			read3FRHeader(ifp, &imagedata, &g3);
			dims[1] = imagedata.height;
			dims[0] = imagedata.width;
			plhs[0] = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
			imagedata.ptr = (float*) mxGetPr(plhs[0]);
			read3FRFloat(ifp, &imagedata, &g3);
			break;

		case HASSELBLAD_DNG:
			readDNGHeader(ifp, &imagedata, &g3);
			dims[1] = imagedata.height;
			dims[0] = imagedata.width;
			plhs[0] = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
			imagedata.ptr = (float*) mxGetPr(plhs[0]);
			readDNGFloat(ifp, &imagedata, &g3);
			break;
	}


	//close the file!
	fclose(ifp);
	
}




//supporting functions


// identification, parsing, exif


void identify(FILE *ifp, globals3fr *g3){
	char head[32], *cp;
	int fsize; //size of the file
	int hlen; //header length
	int i;
	short order; //the byte order (not a stack of pancakes)

	order = get2(ifp, g3); //'II', little endian encoding. (good, this is easy.)
	//note: order is declared as a global variable. here, it's the data order for the basic descriptor info.
	hlen = get4(ifp, g3);
	fseek (ifp, 0, SEEK_SET);
	fread (head, 1, 32, ifp);
	fseek (ifp, 0, SEEK_END);
	fsize = ftell(ifp);

	parse_tiff(ifp, g3, 0);

	cp = g3->make + strlen(g3->make);		/* Remove trailing spaces */
	while (*--cp == ' ') *cp = 0;
	cp = g3->model + strlen(g3->model);
	while (*--cp == ' ') *cp = 0;
	i = (int)strlen(g3->make);			/* Remove make from model */
	if (!_strnicmp (g3->model, g3->make, i) && g3->model[i++] == ' ')
		memmove (g3->model, g3->model+i, 64-i);
	g3->make[63] = 0; //ensure null-termination on the strings.
	g3->model[63] = 0;
	if (!g3->height) g3->height = g3->raw_height;
	if (!g3->width)  g3->width  = g3->raw_width;
	
	//load_raw = &CLASS hasselblad_load_raw; //this changes the earlier load_raw def'n
	/*if (filters == UINT_MAX) filters = 0x94949494;
	if (!cdesc[0])
		strcpy (cdesc, colors == 3 ? "RGB":"GMCY");
	if (filters && colors == 3)
		for (i=0; i < 32; i+=4) {
			if ((filters >> i & 15) == 9)
				filters |= 2 << i;
			if ((filters >> i & 15) == 6)
				filters |= 8 << i;
		}
	*/

	//if (g3->flip == -1) g3->flip = g3->tiff_flip;
	//if (g3->flip == -1) g3->flip = 0;
	g3->filters = 0xb4b4b4b4; //fixed value for 3FR
}




void CLASS parse_tiff (FILE *ifp, globals3fr *g3, int base)
{

	int doff, raw = -1;
	int max_samp = 0;
	int i;

	g3->tiff_nifds = 0; //set the count to zero; this will get incremented as ifds are "discovered".

	fseek (ifp, base, SEEK_SET);
	g3->order = get2(ifp, g3); //the byte-order for the tiff header. ends up being little-endian, 'II'.
	get2(ifp, g3); //advance by another two bytes.
	//memset (g3->tiff_ifd, 0, sizeof g3->tiff_ifd); //set all the ifd entries to zero


	while ((doff = get4(ifp, g3))) {
		fseek (ifp, doff+base, SEEK_SET); //jump to the next ifd offset
		if (parse_tiff_ifd (ifp, g3, base)) break;
	}
	//g3->thumb_misc; //TODO: move global into info struct

	for (i=0; i < g3->tiff_nifds; i++) {
		if (max_samp < g3->tiff_ifd[i].samples)
			max_samp = g3->tiff_ifd[i].samples;
		if (max_samp > 3) max_samp = 3;
		if ((g3->tiff_ifd[i].comp != 6 || g3->tiff_ifd[i].samples != 3) &&
		(g3->tiff_ifd[i].width | g3->tiff_ifd[i].height) < 0x10000 &&
		g3->tiff_ifd[i].width*g3->tiff_ifd[i].height > g3->raw_width*g3->raw_height) {
			  g3->raw_width     = g3->tiff_ifd[i].width;
			  g3->raw_height    = g3->tiff_ifd[i].height;
			  g3->tiff_bps      = g3->tiff_ifd[i].bps;
			  g3->tiff_compress = g3->tiff_ifd[i].comp;
			  g3->data_offset   = g3->tiff_ifd[i].offset;
			  g3->tiff_flip     = g3->tiff_ifd[i].flip;
			  g3->tiff_samples  = g3->tiff_ifd[i].samples;
			  g3->raw = i;
		}
	}

	if (g3->tiff_ifd[0].flip) g3->tiff_flip = g3->tiff_ifd[0].flip;
	
	//set the decoding method -- fixed for 3FR:hasselblad_load_raw.

	//skip the part the sets thumbnail information
}




int CLASS parse_tiff_ifd (FILE *ifp, globals3fr *g3, int base)
{
	unsigned tag, type, len, save, entries;
	int i, ifd;

//	char make[64], model[64]; //does this need to be global, or passed?
	//TODO: add these into the tiffinfo structure.

	ifd = g3->tiff_nifds++; //look at the 0th ifd first.
	//TODO: check this against dcraw! (I think it gets changed w subsequent calls to parse_tiff_ifd?)

	entries = get2(ifp, g3); //number of tags to read in this ifd

	while (entries--) { //step through the entries
		tiff_get (ifp, g3, base, &tag, &type, &len, &save); //read this tiff tag
		switch (tag){
			//handle specific tags...

			case 2: case 256:			// ImageWidth 
				g3->tiff_ifd[ifd].width = getint(ifp, g3, type);
				break;
			case 3: case 257:			// ImageHeight 
				g3->tiff_ifd[ifd].height = getint(ifp, g3, type);
				break;
			case 258:				// BitsPerSample 
				g3->tiff_ifd[ifd].samples = len & 7;
				g3->tiff_ifd[ifd].bps = get2(ifp, g3);
				break;
			case 259:				// Compression 
				g3->tiff_ifd[ifd].comp = get2(ifp, g3);
				break;
			case 262:				// PhotometricInterpretation 
				g3->tiff_ifd[ifd].phint = get2(ifp, g3);
				break;
			case 271:				/* Make */
				fgets (g3->make, 64, ifp);
				break;
			case 272:				/* Model */
				fgets (g3->model, 64, ifp);
				break;
			case 273:				/* StripOffset */
			case 513:
				g3->tiff_ifd[ifd].offset = get4(ifp, g3) + base;
				/*if (!tiff_ifd[ifd].bps) {
				  fseek (ifp, tiff_ifd[ifd].offset, SEEK_SET);
				  if (ljpeg_start (&jh, 1)) {
					tiff_ifd[ifd].comp    = 6;
					tiff_ifd[ifd].width   = jh.wide << (jh.clrs == 2);
					tiff_ifd[ifd].height  = jh.high;
					tiff_ifd[ifd].bps     = jh.bits;
					tiff_ifd[ifd].samples = jh.clrs;
				  }*/ //don't need this?
				break;
			case 274:				/* Orientation */
				g3->tiff_ifd[ifd].flip = "50132467"[get2(ifp, g3) & 7]-'0';
				break;
			case 277:				/* SamplesPerPixel */
				g3->tiff_ifd[ifd].samples = getint(ifp, g3, type) & 7;
				break;
			case 279:				/* StripByteCounts */
			case 514:
				g3->tiff_ifd[ifd].bytes = get4(ifp, g3);
				break;
			case 306:				/* DateTime */
				// TODO: implement the timestamp global
				//into a pass-by-reference structure.
				//get_timestamp(ifp);
				break;
			case 330:
				while (len--) {
				  i = ftell(ifp);
				  fseek (ifp, get4(ifp, g3)+base, SEEK_SET);
				  if (parse_tiff_ifd (ifp, g3, base)) break;
				  fseek (ifp, i+4, SEEK_SET);
				}
				break;
			case 50717:			/* WhiteLevel */
				g3->maximum = getint(ifp, g3, type);
				break;
			case 34665:			/* EXIF tag */
				fseek (ifp, get4(ifp, g3) + base, SEEK_SET);
				parse_exif (ifp, g3, base);
				break;
			//case 50721: color matrix stored into the camera (ignoring right now)
			//case 50728: as-shot neutral level (ignoring right now)
		}

		fseek (ifp, save, SEEK_SET); //go to the next tag.
	}

	//check some color profiles, scaling, junk. ignoring this for now.
	//TODO: double-check that the internal color profiles can be ignored.

	return 0;
}





void CLASS tiff_get (FILE *ifp, globals3fr *g3, unsigned base,
	unsigned *tag, unsigned *type, unsigned *len, unsigned *save)
{
  *tag  = get2(ifp, g3);
  *type = get2(ifp, g3);
  *len  = get4(ifp, g3);
  *save = ftell(ifp) + 4;
  if (*len * ("11124811248488"[*type < 14 ? *type:0]-'0') > 4)
    fseek (ifp, get4(ifp, g3)+base, SEEK_SET);
}




/*
   Since the TIFF DateTime string has no timezone information,
   assume that the camera's clock was set to Universal Time.
 */
void CLASS get_timestamp (FILE *ifp, globals3fr *g3)
{
  struct tm t;
  char str[20];
  //int i;

  str[19] = 0;
  fread (str, 19, 1, ifp);
  memset (&t, 0, sizeof t);
  if (sscanf (str, "%d:%d:%d %d:%d:%d", &t.tm_year, &t.tm_mon,
	&t.tm_mday, &t.tm_hour, &t.tm_min, &t.tm_sec) != 6)
    return; //not enough data in the file to fill in a TIFF timestamp.

  t.tm_year -= 1900; //used for converting the registered day/time into a timestamp, I think. (??)
  t.tm_mon -= 1;
  if (mktime(&t) > 0) //why??
    g3->timestamp = mktime(&t);
}





void CLASS parse_exif (FILE *ifp, globals3fr *g3, int base)
{
  unsigned entries, tag, type, len, save; //, c;
  //double expo;

  entries = get2(ifp, g3); //number of exif tags to read
  while (entries--) {
    tiff_get (ifp, g3, base, &tag, &type, &len, &save);
    switch (tag) {
		case 34855:  g3->iso_speed = get2(ifp, g3);			break;
		case 36867:
		case 36868:  get_timestamp(ifp, g3);			break; //different timestamp? ("Original")
		case 37500:  parse_makernote (ifp, g3, base, 0);		break;
    }
    fseek (ifp, save, SEEK_SET);
  }
}




void CLASS parse_makernote (FILE *ifp, globals3fr *g3, int base, int uptag)
{
	//unsigned offset = 0, c;
	unsigned entries, tag, type, len, save;
	//char buf[10];
	//fread (buf, 1, 10, ifp); //see if we can short-cut it, get a hint... nope.
	//fseek (ifp, -10, SEEK_CUR);
	entries = get2(ifp, g3);
	while (entries--) {
		tiff_get (ifp, g3, base, &tag, &type, &len, &save);
		tag |= uptag << 16;
		if (tag == 0x15 && type == 2) // && is_raw) //is_raw = 1;
			fread (g3->model, 64, 1, ifp);
		fseek (ifp, save, SEEK_SET);
	}
}




void reset_tiff_ifds(globals3fr *g3){
	int i;
	for (i=0; i<MAX_IFDS; i++) {
		g3->tiff_ifd[i].width = 0;
		g3->tiff_ifd[i].height = 0;
		g3->tiff_ifd[i].bps = 0;
		g3->tiff_ifd[i].comp = 0;
		g3->tiff_ifd[i].phint = 0;
		g3->tiff_ifd[i].offset = 0;
		g3->tiff_ifd[i].flip = 0;
		g3->tiff_ifd[i].samples = 0;
		g3->tiff_ifd[i].bytes = 0;
	}
}

void reset_globals(globals3fr *g3){
	reset_tiff_ifds(g3);
	g3->raw_width = 0;
	g3->raw_height = 0;
	g3->height =0;
	g3->width = 0;
	g3->top_margin = 0;
	g3->left_margin = 0;
	g3->order = 0x4949; //big-endian (a guess)
}











// functions for decoding/reading actual data


unsigned CLASS ph1_bithuff (FILE *ifp, globals3fr *g3, int nbits, ushort *huff)
{
  static UINT64 bitbuf=0;
  static int vbits=0;
  unsigned c;

  if (nbits == -1)
    return (unsigned int)(bitbuf = vbits = 0);
  if (nbits == 0) return 0;
  if (vbits < nbits) {
    bitbuf = bitbuf << 32 | get4(ifp, g3);
    vbits += 32;
  }
  c = (unsigned int)(bitbuf << (64-vbits) >> (64-nbits));
  if (huff) {
    vbits -= huff[c] >> 8;
    return (uchar) huff[c];
  }
  vbits -= nbits;
  return c;
}
#define ph1_bits(ifp,g3,n) ph1_bithuff(ifp,g3,n,0)
#define ph1_huff(ifp,g3,h) ph1_bithuff(ifp,g3,*h,h+1)






int CLASS ljpeg_start (FILE *ifp, struct jhead *jh, int info_only)
{
  int c, tag, len;
  uchar data[0x10000];
  const uchar *dp;

	memset (jh, 0, sizeof *jh);
	jh->restart = INT_MAX;
	fread (data, 2, 1, ifp);
	if (data[1] != 0xd8) return 0; //check if it's a valid JPEG file.
	do {
		fread (data, 2, 2, ifp);
		tag =  data[0] << 8 | data[1];
		len = (data[2] << 8 | data[3]) - 2;
		if (tag <= 0xff00) return 0;
		fread (data, 1, len, ifp);
		switch (tag) {
			case 0xffc3:
				jh->sraw = ((data[7] >> 4) * (data[7] & 15) - 1) & 3;
			case 0xffc0:
				jh->bits = data[0];
				jh->high = data[1] << 8 | data[2];
				jh->wide = data[3] << 8 | data[4];
				jh->clrs = data[5] + jh->sraw;
				if (len == 9) getc(ifp);
				break;
			case 0xffc4:
				for (dp = data; dp < data+len && (c = *dp++) < 4; )
				  jh->free[c] = jh->huff[c] = make_decoder_ref (&dp);
				break;
			case 0xffda:
				jh->psv = data[1+data[0]*2];
				jh->bits -= data[3+data[0]*2] & 15;
				break;
			case 0xffdd:
				jh->restart = data[0] << 8 | data[1];
			} //end-switch
	} while (tag != 0xffda); //end-do

	for( c=0; c<5; c++) if (!jh->huff[c+1]) jh->huff[c+1] = jh->huff[c];
	if (jh->sraw) {
		for (c=0; c<4; c++) 
			jh->huff[2+c] = jh->huff[1];
		for (c=0; c < jh->sraw; c++)
			jh->huff[1+c] = jh->huff[0];
	}
	jh->row = (ushort *) calloc (jh->wide*jh->clrs, 4);
	//return g3->zero_after_ff = 1;
	return 1;
}


void CLASS ljpeg_end (struct jhead *jh)
{
	int c;
  for (c=0; c<4; c++) 
	  if (jh->free[c]) 
		  free (jh->free[c]);
  free (jh->row);
}



ushort * CLASS make_decoder_ref (const uchar **source)
{
  int maxv, len, h, i, j;
  const uchar *count;
  ushort *huff;

  count = (*source += 16) - 17;
  for (maxv=16; maxv && !count[maxv]; maxv--);
  huff = (ushort *) calloc (1 + (1 << maxv), sizeof *huff);
  //merror (huff, "make_decoder()");
  huff[0] = maxv;
  for (h=len=1; len <= maxv; len++)
    for (i=0; i < count[len]; i++, ++*source)
      for (j=0; j < 1 << (maxv-len); j++)
	if (h <= 1 << maxv)
	  huff[h++] = len << 8 | **source;
  return huff;
}









// functions for loading 3FR data, either as floats or unsigned shorts

//unsigned shorts
void CLASS hasselblad_load_raw(FILE *ifp, globals3fr *g3, ushort (*image)[4])
{
	struct jhead jh;
	int row, col, pred[2], len[2], diff, c;

	if (!ljpeg_start (ifp, &jh, 0)) return;
	g3->order = 0x4949; //the order is fixed for hassie, apparently.
	ph1_bits(ifp, g3, -1);
	for (row = -g3->top_margin; row < g3->height; row++) {
		pred[0] = pred[1] = 0x8000;
		for (col = -g3->left_margin; col < g3->raw_width - g3->left_margin; col+=2) {
			for (c=0; c<2; c++) len[c] = ph1_huff(ifp,g3,jh.huff[0]);
			for(c=0; c<2; c++) {
				diff = ph1_bits(ifp,g3,len[c]);
				if ((diff & (1 << (len[c]-1))) == 0)
				  diff -= (1 << len[c]) - 1;
				if (diff == 65535) diff = -32768;
				pred[c] += diff;
				if (row >= 0 && (unsigned)(col+c) < g3->width)
					BAYER(row,col+c) = pred[c];
			}
		}
	}
	ljpeg_end (&jh);
	g3->maximum = 0xffff;
}


//floats
void CLASS hasselblad_load_raw_float(FILE *ifp, globals3fr *g3, float *imgPtr)
{
	struct jhead jh;
	int row, col, pred[2], len[2], diff, c, order;
	unsigned short predc;
	//int predval;

	if (!ljpeg_start (ifp, &jh, 0)) return;
	g3->order = 0x4949; //constant for 3frs, _I think_
	ph1_bits(ifp, g3, -1);
	for (row = -g3->top_margin; row < g3->height; row++) {
		pred[0] = pred[1] = 0x8000;
		for (col = -g3->left_margin; col < g3->raw_width - g3->left_margin; col+=2) {
			for (c=0; c<2; c++) 
				len[c] = ph1_huff(ifp, g3, jh.huff[0]);
			for(c=0; c<2; c++) {
				diff = ph1_bits(ifp, g3, len[c]);
				if ((diff & (1 << (len[c]-1))) == 0)
				  diff -= (1 << len[c]) - 1;
				if (diff == 65535) diff = -32768;
				pred[c] += diff;
				if (row >= 0 && (unsigned)(col+c) < g3->width){
					//if (pred[c]<0){
						//do nothing
					//	mexPrintf("(row,col,p,c) = (%i, %i, %i)\n", row, col, pred[c], c);
					//}
					//predc = pred[c];
					//imgPtr[(row) * g3->width + (col+c)] = (float) pred[c]; //don't care about which color channel.
					imgPtr[(row) * g3->width + (col+c)] = (float) ((unsigned short)pred[c]);
				}
			}
		}
	}
	ljpeg_end (&jh);
	g3->maximum = 0xffff;

	//for (int i=0; i<12; i++)
	//	printf("image[%i] : %f\n",i,imgPtr[i]);

}










//functions for reading/decoding Hassie DNG data files as floats


void CLASS adobe_dng_load_raw_lj(FILE *ifp, globals3fr *g3, float *imgPtr)
{
	unsigned save, trow=0, tcol=0, jwide, jrow, jcol, row, col;
	struct jhead jh;
	ushort *rp;

	while (trow < g3->raw_height) {
		save = ftell(ifp);
		if (g3->tile_length < INT_MAX)
			fseek (ifp, get4(ifp, g3), SEEK_SET);
		if (!ljpeg_start_dng (ifp, g3, &jh, 0)) 
			break;
		jwide = jh.wide;
		if (g3->filters) 
			jwide *= jh.clrs;
		//jwide /= is_raw; //wha? is_raw = 1, so this is stupid.
		for (row=col=jrow=0; jrow < jh.high; jrow++) {
			rp = ljpeg_row (jrow, &jh, ifp, g3);
			for (jcol=0; jcol < jwide; jcol++) {
				adobe_copy_pixel (g3, imgPtr, trow+row, tcol+col, &rp);
				if (++col >= g3->tile_width || col >= g3->raw_width)
					row += 1 + (col = 0);
			}
		}
		fseek (ifp, save+4, SEEK_SET);
		if ((tcol += g3->tile_width) >= g3->raw_width)
			trow += g3->tile_length + (tcol = 0);
		ljpeg_end_dng (&jh);
	}
}


//NB: floats!
void CLASS adobe_copy_pixel (globals3fr *g3, float *imgPtr, int row, int col, ushort **rp)
{
	unsigned r, c;

	r = row -= g3->top_margin;
	c = col -= g3->left_margin;

	//if (filters) {

	if (r < g3->height && c < g3->width)
		imgPtr[r*g3->width + c] = (float)((unsigned short)(**rp));
	//BAYER(r,c) = **rp < 0x1000 ? curve[**rp] : **rp;
	//*rp += is_raw;
	*rp += 1; //increment the pointer to the next value, is_raw = 1.
	//side note: need to use += 1 for *rp, *rp ++ doesn't work. (maybe (*rp)++?

	//	} 
	/*else {
		if (r < height && c < width)
		for(c=0; c<1; c++) //tiff_samples = 1
		image[row*width+col][c] = (*rp)[c] < 0x1000 ? curve[(*rp)[c]]:(*rp)[c];
		*rp += tiff_samples;
	}*/
	//if (is_raw == 2 && shot_select) (*rp)--;
}




int CLASS ljpeg_start_dng (FILE *ifp, globals3fr *g3, struct jhead *jh, int info_only)
{
  int c, tag, len;
  uchar data[0x10000];
  const uchar *dp;

	memset (jh, 0, sizeof *jh);
	jh->restart = INT_MAX;
	fread (data, 2, 1, ifp);
	if (data[1] != 0xd8) return 0; //check if it's a valid JPEG file.
	do {
		fread (data, 2, 2, ifp);
		tag =  data[0] << 8 | data[1];
		len = (data[2] << 8 | data[3]) - 2;
		if (tag <= 0xff00) return 0;
		fread (data, 1, len, ifp);
		switch (tag) {
			case 0xffc3:
				jh->sraw = ((data[7] >> 4) * (data[7] & 15) - 1) & 3;
			case 0xffc0:
				jh->bits = data[0];
				jh->high = data[1] << 8 | data[2];
				jh->wide = data[3] << 8 | data[4];
				jh->clrs = data[5] + jh->sraw;
				if (len == 9 && !g3->dng_version) getc(ifp);
				break;
			case 0xffc4:
				for (dp = data; dp < data+len && (c = *dp++) < 4; )
				  jh->free[c] = jh->huff[c] = make_decoder_ref_dng (&dp);
				break;
			case 0xffda:
				jh->psv = data[1+data[0]*2];
				jh->bits -= data[3+data[0]*2] & 15;
				break;
			case 0xffdd:
				jh->restart = data[0] << 8 | data[1];
			} //end-switch
	} while (tag != 0xffda); //end-do

	for( c=0; c<5; c++) if (!jh->huff[c+1]) jh->huff[c+1] = jh->huff[c];
	if (jh->sraw) {
		for (c=0; c<4; c++) 
			jh->huff[2+c] = jh->huff[1];
		for (c=0; c < jh->sraw; c++)
			jh->huff[1+c] = jh->huff[0];
	}
	jh->row = (ushort *) calloc (jh->wide*jh->clrs, 4);
	//return g3->zero_after_ff = 1;
	return 1;
}




void CLASS ljpeg_end_dng (struct jhead *jh)
{
	int c;
  for (c=0; c<4; c++) 
	  if (jh->free[c]) 
		  free (jh->free[c]);
  free (jh->row);
}




ushort * CLASS make_decoder_ref_dng (const uchar **source)
{
  int maxv, len, h, i, j;
  const uchar *count;
  ushort *huff;

  count = (*source += 16) - 17;
  for (maxv=16; maxv && !count[maxv]; maxv--);
  huff = (ushort *) calloc (1 + (1 << maxv), sizeof *huff);
  //merror (huff, "make_decoder()");
  huff[0] = maxv;
  for (h=len=1; len <= maxv; len++)
    for (i=0; i < count[len]; i++, ++*source)
      for (j=0; j < 1 << (maxv-len); j++)
	if (h <= 1 << maxv)
	  huff[h++] = len << 8 | **source;
  return huff;
}






ushort * CLASS ljpeg_row (int jrow, struct jhead *jh, FILE *ifp, globals3fr *g3)
{
	int col, c, diff, pred, spred=0;
	ushort mark=0, *row[3];

	if (jrow * jh->wide % jh->restart == 0) {
		for (c=0; c<6; c++)
			jh->vpred[c] = 1 << (jh->bits-1);
		if (jrow) {
			fseek (ifp, -2, SEEK_CUR);
			do mark = (mark << 8) + (c = fgetc(ifp));
			while (c != EOF && mark >> 4 != 0xffd);
		}
		getbits(ifp,g3,-1);
	}
	for (c=0; c<3; c++)
		row[c] = jh->row + jh->wide*jh->clrs*((jrow+c) & 1);
	for (col=0; col < jh->wide; col++) {
		for(c=0; c<jh->clrs; c++) {
			diff = ljpeg_diff (ifp, g3, jh->huff[c]);
			if (jh->sraw && c <= jh->sraw && (col | c))
				pred = spred;
			else if (col) 
				pred = row[0][-jh->clrs];
			else	    
				pred = (jh->vpred[c] += diff) - diff;
			if (jrow && col) 
				switch (jh->psv) {
					case 1:	break;
					case 2: pred = row[1][0];					break;
					case 3: pred = row[1][-jh->clrs];				break;
					case 4: pred = pred +   row[1][0] - row[1][-jh->clrs];		break;
					case 5: pred = pred + ((row[1][0] - row[1][-jh->clrs]) >> 1);	break;
					case 6: pred = row[1][0] + ((pred - row[1][-jh->clrs]) >> 1);	break;
					case 7: pred = (pred + row[1][0]) >> 1;				break;
					default: pred = 0;
				}
			if ((**row = pred + diff) >> jh->bits) { }
			//	derror(); //unexpected end-of-file.
			if (c <= jh->sraw) 
				spred = **row;
			row[0]++; row[1]++;
		}
	}
	return row[2];
}




int CLASS ljpeg_diff (FILE *ifp, globals3fr *g3, ushort *huff)
{
  int len, diff;

  len = gethuff(ifp, g3, huff);
  if (len == 16 && (!g3->dng_version || g3->dng_version >= 0x1010000))
    return -32768;
  diff = getbits(ifp, g3, len);
  if ((diff & (1 << (len-1))) == 0)
    diff -= (1 << len) - 1;
  return diff;
}


/*
   getbits(-1) initializes the buffer
   getbits(n) where 0 <= n <= 25 returns an n-bit integer
 */
unsigned CLASS getbithuff (FILE *ifp, globals3fr *g3, int nbits, ushort *huff)
{
  static unsigned bitbuf=0;
  static int vbits=0, reset=0;
  unsigned c;

  if (nbits == -1)
    return bitbuf = vbits = reset = 0;
  if (nbits == 0 || vbits < 0) return 0;
  while (!reset && vbits < nbits && (c = fgetc(ifp)) != EOF &&
    !(reset = g3->zero_after_ff && c == 0xff && fgetc(ifp))) {
    bitbuf = (bitbuf << 8) + (uchar) c;
    vbits += 8;
  }
  c = bitbuf << (32-vbits) >> (32-nbits);
  if (huff) {
    vbits -= huff[c] >> 8;
    c = (uchar) huff[c];
  } else
    vbits -= nbits;
//  if (vbits < 0) derror();
  return c;
}

//#define getbits(ifp,g3,n) getbithuff(ifp,g3,n,0)
//#define gethuff(ifp,g3,h) getbithuff(ifp,g3,*h,h+1)






// utility functions for identifying and reading exif, tiff tags from a DNG file


void CLASS identifyDNG(globals3fr *g3, FILE *ifp){
	char *cp;
	int i;

	g3->order = get2_dng(ifp);
	parse_tiff_dng(0, g3, ifp);

	cp = g3->make + strlen(g3->make);		/* Remove trailing spaces */
	while (*--cp == ' ') *cp = 0;
	cp = g3->model + strlen(g3->model);
	while (*--cp == ' ') *cp = 0;
	i = (int)strlen(g3->make);			/* Remove make from model */
	if (!_strnicmp (g3->model, g3->make, i) && g3->model[i++] == ' ')
		memmove (g3->model, g3->model+i, 64-i);
	g3->make[63] = 0; //ensure null-termination on the strings.
	g3->model[63] = 0;

	if (!g3->height) g3->height = g3->raw_height;
	if (!g3->width)  g3->width  = g3->raw_width;

	if (!g3->cdesc[0])
		strcpy (g3->cdesc, "RGB"); //for fun, not necessary.

	//do some mangling of the filter value to get:
	g3->filters = 0xb4b4b4b4; //...the same as the 3fr. good.

}


void CLASS parse_tiff_dng(int base, globals3fr *g3, FILE *ifp){
	int doff, raw=-1;
	int max_samp = 0;
	int i;

	g3->tiff_nifds = 0; //reset the number of ifds

	fseek (ifp, base, SEEK_SET);
	g3->order = get2(ifp, g3); // should be 0x4d4d = II for little-endians or 0x4949 = MM for Big-endians
	get2(ifp, g3); //advance two bytes
	
	while ((doff = get4(ifp, g3))) {
		fseek (ifp, doff+base, SEEK_SET); //jump to the next ifd offset
		if (parse_tiff_ifd_dng (ifp, g3, base)) break;
	}

	//skipping the thumbnail info.

	//copy the info from the main data image's ifd into g3's "global" vars.
	for (i=0; i < g3->tiff_nifds; i++) {
		if (max_samp < g3->tiff_ifd[i].samples)
			max_samp = g3->tiff_ifd[i].samples;
		if (max_samp > 3) max_samp = 3;
		if ((g3->tiff_ifd[i].comp != 6 || g3->tiff_ifd[i].samples != 3) &&
		(g3->tiff_ifd[i].width | g3->tiff_ifd[i].height) < 0x10000 &&
		g3->tiff_ifd[i].width*g3->tiff_ifd[i].height > g3->raw_width*g3->raw_height) {
			  g3->raw_width     = g3->tiff_ifd[i].width;
			  g3->raw_height    = g3->tiff_ifd[i].height;
			  g3->tiff_bps      = g3->tiff_ifd[i].bps;
			  g3->tiff_compress = g3->tiff_ifd[i].comp;
			  g3->data_offset   = g3->tiff_ifd[i].offset;
			  g3->tiff_flip     = g3->tiff_ifd[i].flip;
			  g3->tiff_samples  = g3->tiff_ifd[i].samples;
			  g3->raw = i;
		}
	}
	
	if (g3->tiff_ifd[0].flip) g3->tiff_flip = g3->tiff_ifd[0].flip;
	
	//compression = 7, so:
	//load_raw = &CLASS lossless_jpeg_load_raw;
	//(this gets changed later, in identify(). )

	//skip some more thumbnail stuff.


}




int CLASS parse_tiff_ifd_dng (FILE *ifp, globals3fr *g3, int base)
{
	unsigned tag, type, len, save, plen=16, entries;
	uchar cfa_pat[16], cfa_pc[] = { 0,1,2,3 }, tab[256];
	int c, i, cfa, ifd;

	ifd = g3->tiff_nifds++; //look at the 0th ifd first, then increment the ifd count.
	entries = get2(ifp, g3); //number of tags to read in this ifd
	
	while (entries--) { //step through the entries
		tiff_get(ifp, g3, base, &tag, &type, &len, &save); //read this tiff tag
		switch (tag){

			case 2: case 256:			// ImageWidth 
				g3->tiff_ifd[ifd].width = getint(ifp, g3, type);
				break;
			case 3: case 257:			// ImageHeight 
				g3->tiff_ifd[ifd].height = getint(ifp, g3, type);
				break;
			case 258:				// BitsPerSample 
				g3->tiff_ifd[ifd].samples = len & 7;
				g3->tiff_ifd[ifd].bps = get2(ifp, g3);
				break;
			case 259:				// Compression 
				g3->tiff_ifd[ifd].comp = get2(ifp, g3);
				break;
			case 262:				// PhotometricInterpretation 
				g3->tiff_ifd[ifd].phint = get2(ifp, g3);
				break;
			case 270:				/* ImageDescription */
				fread (g3->desc, 512, 1, ifp);
				break;
			case 271:				/* Make */
				fgets (g3->make, 64, ifp);
				break;
			case 272:				/* Model */
				fgets (g3->model, 64, ifp);
				break;
			case 273:				/* StripOffset */
			case 513:
				g3->tiff_ifd[ifd].offset = get4(ifp, g3) + base;
				//skip a logical test which gives a false.
				break;
			case 274:				/* Orientation */
				g3->tiff_ifd[ifd].flip = "50132467"[get2(ifp, g3) & 7]-'0';
				break;
			case 277:				/* SamplesPerPixel */
				g3->tiff_ifd[ifd].samples = getint(ifp, g3, type) & 7;
				break;
			case 279:				/* StripByteCounts */
			case 514:
				g3->tiff_ifd[ifd].bytes = get4(ifp, g3);
				break;
			case 306:				/* DateTime */
				//get_timestamp_dng(ifp,g3);
				//there's a better timestamp stored in the EXIF data, use that instead.
				break;
			case 330:				//sub ifd's
				while (len--) {
				  i = ftell(ifp);
				  fseek (ifp, get4(ifp, g3)+base, SEEK_SET);
				  if (parse_tiff_ifd_dng (ifp, g3, base)) break;
				  fseek (ifp, i+4, SEEK_SET);
				}
				break;
			case 322:				/* TileWidth */
				g3->tile_width = getint(ifp, g3, type);
				break;
			case 323:				/* TileLength */
				g3->tile_length = getint(ifp, g3, type);
				break;
			case 33422:		//CFA pattern
				if ((plen=len) > 16) plen = 16;
				fread (cfa_pat, 1, plen, ifp);
				for (g3->colors=cfa=i=0; i < plen; i++) {
					g3->colors += !(cfa & (1 << cfa_pat[i]));
					cfa |= 1 << cfa_pat[i];
				} //at the end: colors = 3, cfa = 7.
				for(c=0; c<g3->colors; c++){
					tab[cfa_pc[c]] = c;
				}
				g3->cdesc[c] = 0;
				for (i=16; i--; )
					g3->filters = g3->filters << 2 | tab[cfa_pat[i % plen]];
				break; //leaves filters = 0x94949494.
			case 50717:			/* WhiteLevel */
				g3->maximum = getint(ifp, g3, type);
				break;
			case 34665:			/* EXIF tag */
				fseek (ifp, get4(ifp, g3) + base, SEEK_SET);
				parse_exif (ifp, g3, base);
				break;
			case 50706:			/* DNGVersion */
				for(c=0; c<4; c++)
					g3->dng_version = (g3->dng_version << 8) + fgetc(ifp);
				//results in dng_version = 16777216 = 0x01000000.
				//is_raw = 1; //duh.
				break;
			//case 50721,50722: color matrix. sets cm[][] values.
			//i want raw data, so i'm skipping this.
			//case 50728: as shot neutral, ignoring right now.

		}

		fseek (ifp, save, SEEK_SET); //move on to the next tag.
	}
	//do some ops with cc, pre_mult... options which we won't be using. skip this part.

	return 0;
}





void reset_globalsDNG(globals3fr *g3){
	reset_tiff_ifds_dng(g3);
	g3->raw_width = 0;
	g3->raw_height = 0;
	g3->height =0;
	g3->width = 0;
	g3->top_margin = 0;
	g3->left_margin = 0;
	g3->dng_version = 0;
	g3->tile_width = INT_MAX;
	g3->tile_length = INT_MAX;
	g3->zero_after_ff = 1;
	g3->order = 0x4d4d; //assume little-endian until we read the file
}


void reset_tiff_ifds_dng(globals3fr *g3){
	int i;
	for (i=0; i<MAX_IFDS; i++) {
		g3->tiff_ifd[i].width = 0;
		g3->tiff_ifd[i].height = 0;
		g3->tiff_ifd[i].bps = 0;
		g3->tiff_ifd[i].comp = 0;
		g3->tiff_ifd[i].phint = 0;
		g3->tiff_ifd[i].offset = 0;
		g3->tiff_ifd[i].flip = 0;
		g3->tiff_ifd[i].samples = 0;
		g3->tiff_ifd[i].bytes = 0;
	}
}














/* **************************************
//
//            .3FR IMAGES 
//          (Hasselblad Raw)
//
// **************************************
*/
//uses a ton of functions: migrated to hasselblad_load_v2.cpp.
//

bool read3FRHeader(FILE *ifp, imageDataStruct *image, globals3fr *g3){
	reset_globals(g3);
	identify(ifp, g3);
	image->width = g3->width;
	image->height = g3->height;
	image->bpp = (int)g3->tiff_bps; //bits per sample... tiff_samples = 1 for grayscale, 3 for rgb, etc
	image->comp = (int)g3->tiff_compress;
	image->byteOffset = (size_t)g3->data_offset;

	return true; //TODO: include more error checking.
}



bool read3FRFloat(FILE *ifp, imageDataStruct *image, globals3fr *g3){

	//image->ptr = (float*) calloc( image->width*image->height, sizeof(float) );
	//could also malloc this one... calloc is marginally cleaner. is malloc any faster?

	fseek (ifp, g3->data_offset, SEEK_SET);
	hasselblad_load_raw_float(ifp, g3, image->ptr);
	if (ferror(ifp)) {
		image->error = "Unknown error while reading 3FR file.";
		return false;
	}

	return true;
}



/* **************************************
//
//            .DNG IMAGES 
//       (Hasselblad DNG-specific,
//          not all DNGs yet.)
//
// **************************************
*/
// functions for DNG reading are in readDNG.h and readDNG.cpp.
//

bool readDNGHeader(FILE *ifp, imageDataStruct *image, globals3fr *g3){
	reset_globalsDNG(g3);
	identifyDNG(g3, ifp);
	image->width = g3->width;
	image->height = g3->height;
	image->bpp = (int)g3->tiff_bps; //bits per sample... tiff_samples = 1 for grayscale, 3 for rgb, etc
	image->comp = (int)g3->tiff_compress;
	image->byteOffset = (size_t)g3->data_offset;

	return true; //TODO: include more error checking.
}



bool readDNGFloat(FILE *ifp, imageDataStruct *image, globals3fr *g3){

	//image->ptr = (float*) calloc( image->width*image->height, sizeof(float) );
	//could also malloc this one... calloc is marginally cleaner. is malloc any faster?

	fseek (ifp, g3->data_offset, SEEK_SET);
	adobe_dng_load_raw_lj(ifp, g3, image->ptr);
	if (ferror(ifp)) {
		image->error = "Unknown error while reading DNG file.";
		return false;
	}

	return true;
}


