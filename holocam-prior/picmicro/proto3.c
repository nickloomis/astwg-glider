//Program to control the prototype DHI unit operations.
//
//NB: This version (2) differs from the original (PROTO1.C) in its use of the TRIS
//port to control the camera.  The camera expects the input pins to be shorted (ie, 
//connected together through ground) when operating, and NC when not.  Instead of
//applying a logic voltage, which keeps the pins from "feeling" shorted, this version
//uses the high impedence of the input TRIS to make the camera think that it's
//disconnected without actually applying a voltage.  This is probably safer for the
//camera, since we've got no idea what's on the other side of the input.  (It should
//be able to handle TTL logic, but we're not sure...)
//
//This version also differs from PROTO2.C by using RB3 as the camera trigger port, one
//of the few pins not connected in the original circuit (and therefor easy to test and 
//solder in new connections). The TRIS's were also checked here... might be erroneous
//in PROTO2. Simple functions now encapsulate turning on and off the camera and laser,
//so that the same instructions are followed every time. Finally, #defines are used to
//create global constants that can be easily modified for the particular data mission that
//the camera is used for.
//
//
//Overview: The PIC determines which auxiliary units are operational at any one time
//depending on the inputs it reads from any number of sense pins.  When external power
//is available (through the Firewire connection), only the camera is powered up.  If
//internal battery power is used, the PIC begins a data sequence, then powers the unit
//down afterwards.
//
//Flow--
//1) Pause for startup and stabilization (a few milliseconds)
// A) Initialize all the inputs and outputs
// B) Status check (or quick signal to the output that the chip's working right)
//2) Check status of power source
// A) Power on the camera only (Firewire power supplied)
//  1. Put the status on an output bit (steady lit)
// B) Begin data sequence... (battery power only) -- (feedback??)
//  1. Turn on camera
//  2. Wait a few seconds
//  3. Power on laser, take 20 frames
//  4. Cycle camera power (reset any errors which might have occured)
//  5. 20 frames, etc., up to the maximum storage capacity of the camera
//  6. Power down camera and laser
//  7. Put the status on an output bit (slow blink)
// C) Programming sequence: overpowers the chip ops, rewrites the code
//
//
//IO Lines
//
// RB0 - Laser trigger
// RB1 - Camera trigger
// RB2 - External shutter (not in current system)
// RB6 - Status1
// RB7 - Status2
// RA0 - Camera Power
// RA1 - Laser Power
// RA4 - Power Sense
// (Other IOs not implemented)
//
//Nick Loomis
//MIT 3D Optical Systems Group
//Underwater DHI project
//February 2008
//
// Change log:
//  2008 - 2010: proto1, proto2, proto3 developed for research work (nloomis@)
//  2010/06/XX -- modified for oil droplet work (nloomis@)
//  2015/12/11 -- clarifying comments added; changelog started (nloomis@)

// sampling/instrument settings
//#define DATA_CAPTURE_DELAY_S 300 // seconds between first capture and start of the data sequence
#define DATA_CAPTURE_DELAY_S 600 // about 12 minutes
//#define CAPTURE_DELAY_MS 7000 // total time between consecutive captures; must be greater than LASER_WARMUP_TIME_MS
// was set to 7000ms for the first five Jack Fitz dives
#define CAPTURE_DELAY_MS 9000 //delay starting at dive six (the second 1-km dive)
#define LASER_WARMUP_DELAY_MS 500  // time for laser controller to stabilize
#define LASER_COOLDOWN_DELAY_MS 15 // time for laser controller to stop working
#define NFRAMES 1500 // total number of frames to capture (about 125 per 8Gb for the CFV-39)
#define NFRAMES_RESET 40 // number of frames between resets of the camera
#define STARTUP_DELAY_S 8 // delay time for camera to turn on, in seconds
#define POWERDOWN_DELAY_S 4 // delay time for camera to write out all the data

// control signals which should be sent to the ports to get some particular action
#define LASER_ON 0
#define LASER_OFF 1
#define LASER_CONTROLLER_ON 1
#define LASER_CONTROLLER_OFF 0
#define CAMERA_POWER_ON 1
#define CAMERA_POWER_OFF 0
// use the following two lines for TRISB triggering (incorrect)
#define CAMERA_EXPOSE_ENABLE 0 //all TRISB as outputs (low impedance + voltage control)
#define CAMERA_EXPOSE_DISABLE 10 //the camera TRISB's as inputs (high impedance, no voltage control)
// or the following two lines for RB1 voltage triggering (correct)
#define CAMERA_EXPOSE 0
#define CAMERA_NO_EXPOSE 1
#define STATUS2_ON 1
#define STATUS2_OFF 0

#include <pic.h>
//#include <pic16f6x.h> //not needed for compiling through the MPLAB IDE

__CONFIG(0x3F02);  //No code or memory protection, tie MCLR to VDD, HS XTAL (use 3D10 for internal RC oscillator)
// use 0x3FF0 for internal oscillator; also set RB6,RB7 to zero in that case

// function prototypes
void laserpulse();
void capturedata(int framecount);
void cameraonly();

void turnoncamera();
void turnoffcamera();
void turnonlaser();
void turnofflaser();

void blinkstatus2(long timeon, long timeoff);

void delaytimer(long time);
void delayseconds(long time);
void qmstimer();
void dit();
void dot();
void letters();

//function definitions

void main(void) {
	PORTA = 0x00;  //all off to start with
	PORTB = 0x00;
	CMCON = 0x07;  //releases RA0-3 for general inputs
	
	//TRISA = 0b00000110; //RA2: PowerSense, RA1: LaserPower, RA0: CameraPower
	TRISA = 0b00000100;
	TRISB = 0b00001010; //RB0: LaserTrig, RB1: CameraTrig, RB2: ShutterTrig, RB3: CameraTrig2, RB6: Status1, RB7: Status2
	//set RB3, RB1 to a high-impedance input, others as outputs or don't-cares

	/*
	//initialize the IO lines
	RB0=0; //control signal to laser; off while powered down
	RB1=0; //camera trigger; ok to be LOW while not powered
	RB3=0; //no power to the camera trigger_2
	RA1=0; //controller power is off
	RA0=0; //camera power is off
	RB6=0; //turn off any status bits
	RB7=0;	
	*/ //done through PORT calls, now, see above.
	
	//initial pause
	turnonlaser();
	letters();	//output...let us know that the system is turned on and the PIC is working
	turnofflaser();

	
	//sense and branch...
	if (RA2) {
		// Unit is supplied power over Firewire, sensed on RA2. In this case, the
        // camera is plugged into the computer, and the micro doesn't need to
        // do anything.
		cameraonly(); //turn on camera for data transfer
	}

	else {
		// If RA2 is low, power is being supplied by the instrument. This is a clear
        // signal that the micro should start a data capture sequence.

        // Pulse the laser using a fixed pattern to let the user know that the micro
        // is responding, and in data gathering mode. (No other indicators were
        // visible from the outside of the instrument.)
		turnonlaser();
		letters();
		letters();
		turnofflaser();

		// Capture a single image on the camera. This is used to set the timing: the
        // camera didn't have a good internal clock, and there was no RTC on the micro.
        // By noting when the first picture was captured, the time in the EXIF tags of
        // the other pictures could be back-corrected.
		turnoncamera();
		RB1 = CAMERA_EXPOSE;
		delaytimer(250); //wait for a short exposure 
		RB1 = CAMERA_NO_EXPOSE;
		delaytimer(2500); //write out the frame
		turnoffcamera();

		// Pause for several minutes to allow the user to deploy the instrument.
        // Following the DATA_CAPTURE_DELAY_S time, the micro moves into the
        // primary data capturing loop.
        // The DATA_CAPTURE_DELAY_S also served as a "safe" period to turn off
        // the instrument again without leaving the camera in a bad state.        
		delayseconds(DATA_CAPTURE_DELAY_S); //some time to reprogram before we start jilting everything around
		capturedata(NFRAMES); //start with a limited number of frames of data! change before sending proto into the wild.
	}

}


void cameraonly(){
	// turns on the camera for transferring data over Firewire.  The system turns off
	// once the Firewire is unplugged.  For the heck of things, this funtion also 
	// sends a final "shut-down" call to the camera power controller if it senses that
	// the Firewire has been disconnected and there's still power from somewhere (ie,
	// a mistakenly plugged in battery...?)

	RA0 = CAMERA_POWER_ON; //turn on power to camera
	RB7 = STATUS2_ON; // Light status2 solid (this LED was visible when the instrument
                      // was disassembled to connect Firewire to the camera) 

	while (RA2) {
		delaytimer(3000);
	}
	RA0 = CAMERA_POWER_OFF; //turn off camera once power is disconnected (not really necessary, I'm thinking)
	RB7 = STATUS2_OFF; //turn off status2
}

// Capture a number of images.

// The data capture sequence controls both the laser and the camera from the micro:
//  1) Turn on power to the laser
//  2) Signal the camera to begin an exposure (in Bulb mode)
//  3) Wait 0.25 ms for the camera to respond and begin exposing
//  4) Pulse the laser for a brief moment
//  5) Remove the exposure signal from the camera
//  6) Turn off power to the laser
//  7) Wait a fixed time for the camera to write its data to disk
//     and any additional time for moving to the next sample location
//
// The camera was not robust. Every NFRAMES_RESET attempted captures, the
// micro would reboot the camera automatically to clear out any errors
// which might have occurred. Once the camera came back online (based on
// a fixed wait time for reboot), the micro resumed its operation.
//
// Status LEDs were built onto the board for debugging operation when
// the instrument was disassembled.
void capturedata(int framecount){
	long fcnt;
	int resetcnt=0;
	int maxreset = NFRAMES_RESET; //number of frames between power cycling the camera
	long msframedelay = CAPTURE_DELAY_MS - LASER_WARMUP_DELAY_MS; //wait time between frames, minus the time required to get ready for a frame
	long turnontime = STARTUP_DELAY_S*1000;
	
	//turn on camera
	turnoncamera();

	//loop over the total number of frames to capture
	for (fcnt = 0; fcnt < framecount; fcnt++){
		//capture a frame of data
		RB7 = STATUS2_ON; //turn on status2

		//RB2=1; //open the shutter (if there is one -- then add a delay...)
		turnonlaser();
		RB1 = CAMERA_EXPOSE;
		qmstimer(); //short time for camera to respond
		laserpulse();
		//RB2=0; //stop the external shutter
		//qmstimer(); //extra short pause
		//delaytimer(2000); //purely for debugging, check to see that the impedance/voltage goes to what we want.
		RB1 = CAMERA_NO_EXPOSE; //stop the camera's exposure

		turnofflaser();
		RB7 = STATUS2_OFF; //turn off status2

		delaytimer(msframedelay); //write out the data, wait for the next data to start

		resetcnt++;
		if (resetcnt > maxreset) {
			delaytimer(1000); //write out the last bit of data, just to be sure
			turnoffcamera();
			delaytimer(1000); //wait for everything to calm down
			turnoncamera();
			resetcnt=0; //reset the counter
		} //back to work.
	}

	//end of data collection
	delaytimer(POWERDOWN_DELAY_S*1000); //extra couple seconds to write out data 
	turnofflaser();
	turnoffcamera();

	blinkstatus2(350, 150); //
}

// Set the micro registers and turn on power to the camera.
void turnoncamera(){
	TRISB = CAMERA_EXPOSE_ENABLE;
	RA0 = CAMERA_POWER_ON;
	RB1 = CAMERA_NO_EXPOSE;
	delaytimer(STARTUP_DELAY_S*1000); //turn on delay
}

// Turn off power to the camera.
void turnoffcamera(){
	//RB1 = CAMERA_NO_EXPOSE:
	RA0 = CAMERA_POWER_OFF;
	TRISB = CAMERA_EXPOSE_DISABLE;
	RB1 = 0; //make sure there's no voltage applied to the camera when it's shut off
}

// Turn on power to the laser diode driver.
void turnonlaser(){
	RB0 = LASER_OFF;
	RA1 = LASER_CONTROLLER_ON;
	delaytimer(LASER_WARMUP_DELAY_MS); //warm up time for laser controller
}

// Turn off power to the laser diode driver.
void turnofflaser(){
	RB0 = LASER_OFF;
	RA1 = LASER_CONTROLLER_OFF;
	delaytimer(LASER_COOLDOWN_DELAY_MS); //time for the controller to stop operating
	RB0 = 0; //no voltage going to the laser
}

// Pulse the laser for a short time.
// NL: I was having trouble figuring out a way to do a short, programmable delay. The
// dummy=X statements each take a fixed time to run (I want to say it was ~2us per
// assignment with a 4Mhz clock speed), so these act like a controlled delay timer.
// As the comments note, there's probably a better way to do this with interrupts.
void laserpulse(){
	int dummy;
	RB0 = LASER_ON; //turn on laser
	dummy=2; //start arbitrary short delay; there's a better way to do this, but I've forgotten, dang.
	dummy=3;
	dummy=4;
	dummy=5;
	dummy=6;
	dummy=7;
	dummy=8;
	dummy=9;
	dummy=10;
	dummy=11;
	dummy=12; //moving in from 16" in 13" changes intensity by I2/I1 = (r1/r2)^2*exp(-C*(r2-r1))= 2.0,
	dummy=13; //where r is the distance, and C is the beam scattering coef (assumed to be 0.1)
	dummy=14;
	//dummy=15; //was at dummy15 for 2010-06-11 oil and air tests and previous
	//dummy=16;
	//dummy=17;
	RB0 = LASER_OFF; //turn off laser
}

// Blinks an LED forever to let the user know that the data collection is finished.
// This function is never exited.
void blinkstatus2(long timeon, long timeoff){
	//dead-end function; blinks the status2 line to letcha know it's done with data.
	while(1){
		RB7 = STATUS2_ON;
		delaytimer(timeon);
		RB7 = STATUS2_OFF;
		delaytimer(timeoff);
	}
}

// Delay operation for a fixed number of milliseconds.
void delaytimer(long time){
	//function to do millisecond delays (approximately)
	long i;
	for (i=0; i<4*time; i++){
		qmstimer();
	}
}

// Delay for a quarter of a millisecond.
// The TMR0 register is incremented reguarly by the clock, and has a fixed size. (NL: I
// seem to recall it's a 0xFF max, or less?) The while() loops until the overflow
// interrupt, T01F, goes true.
//
// NL: during the development cycle, the chip's timing moved from a 4Mhz clock to an 8Mhz
// clock. Being lazy and under a time crunch, I copied the timer loop twice.
// I'm not 100% sure why I didn't use a similar timer for the laser pulse; it may have
// been too coarse of a resolution, but I'm unsure now.)
void qmstimer(){
	T0CS=0;
	T0IE=0;
	TMR0=0x15; //for internal 4 MHz operation
	T0IF=0;
	while(!T0IF){
	}
	//loop it again, we're at 8 MHz now instead of 4, and the timer doesn't have enough
	//empty counts to do a full 250us count -- only 125.  We might also expect a faster
	//return from the function, so we can compensate by tacking on a few extra counts.
	TMR0=0x15;
	T0IF=0;
	while(!T0IF){
	}
}

// Delay for a fixed number of seconds. A status LED is used to indicate the
// delay cycle.
void delayseconds(long time){
	long i;
	for (i=0; i<time; i++){
		RB7 = STATUS2_ON;
		delaytimer(50);
		RB7 = STATUS2_OFF;
		delaytimer(945); //total of one second wait
	}
}

// Pulse the laser with a Morse code "dit". Part of an operation to provide
// unique signals to the user.
void dit(){
	RB7 = STATUS2_ON;
	RB0 = LASER_ON;
	delaytimer(100);
	RB7 = STATUS2_OFF;
	RB0 = LASER_OFF;
	delaytimer(100);
}

// Pulse the laser with a Morse code "dot". Part of an operation to provide
// unique signals to the user.
void dot(){
	RB7 = STATUS2_ON;
	RB0 = LASER_ON;
	delaytimer(250);
	RB7 = STATUS2_OFF;
	RB0 = LASER_OFF;
	delaytimer(100);
}

// Pulse the laser to display "DH" in Morse code. This was a signal to
// the user that the micro was in data-capture mode, and the only visible
// signal once the instrument was assembled.
void letters(){
	//D
	dot();
	dit();
	dit();
	delaytimer(100);
	//H
	dit();
	dit();
	dit();
	dit();
	delaytimer(100);
}
