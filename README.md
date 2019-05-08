This is a minimal (but usable) example of the SDR interface the Linux Kernel has, without relying on large external projects or complex libraries.

The FM decoder pass is written in SSE3/AVX2 including a relatively accurate and generalized vectorized ATAN2 function which may also be of interest to others. Generally the USB subsystem uses more CPU than this project does when playing FM radio.

Feel free to re-use any and all of this code for any project you wish. I chose the MIT license for a reason, afterall.

Usage is very simple: The program accepts a single argument, the frequency (in the standard I'm used to here in the US of "102.1" for example) and attempts to autodetect and find the appropriate SDR dongle. It was tested with the RTL-SDR dongle, and sits in the sub-1% CPU usage when running.
